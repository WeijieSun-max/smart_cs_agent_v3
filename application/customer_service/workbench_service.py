from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from threading import RLock
from typing import Any

from domain.customer_service_agent.service import conversation_archive_service, short_term_memory_service
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from pkg.config.settings import ENV_FILE, get_settings
from pkg.security import get_local_user_id, set_local_user_id

_user_settings_lock = RLock()

EXCLUDED_NAMES = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "dist",
    "logs",
    "node_modules",
    "vector_store",
}

LANGUAGES = {
    ".css": "css",
    ".html": "html",
    ".js": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def get_current_user() -> dict[str, str]:
    return {"user_id": get_local_user_id()}


def update_current_user(user_id: str) -> dict[str, str]:
    with _user_settings_lock:
        _persist_local_user_id(user_id)
        _retarget_user_scoped_services(user_id)
        set_local_user_id(user_id)
    return {"user_id": user_id}


def _persist_local_user_id(user_id: str) -> None:
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    replacement = f"LOCAL_USER_ID={user_id}"
    updated: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith("LOCAL_USER_ID="):
            if not replaced:
                updated.append(replacement)
                replaced = True
            continue
        updated.append(line)
    if not replaced:
        updated.append(replacement)
    temporary = ENV_FILE.with_suffix(ENV_FILE.suffix + ".tmp")
    temporary.write_text("\n".join(updated) + "\n", encoding="utf-8")
    os.replace(temporary, ENV_FILE)


def _retarget_user_scoped_services(user_id: str) -> None:
    targets: list[object] = []
    try:
        memory = short_term_memory_service.get_service().memory
        targets.extend([memory, getattr(memory, "cache", None), getattr(memory, "archive", None)])
    except RuntimeError:
        pass
    archive = conversation_archive_service.get_service_or_none()
    if archive is not None:
        targets.append(archive)
    for target in targets:
        if target is None:
            continue
        switch_user = getattr(target, "switch_user", None)
        if callable(switch_user):
            switch_user(user_id)
        elif hasattr(target, "user_id"):
            target.user_id = user_id


def list_agents() -> list[dict[str, Any]]:
    settings = get_settings()
    model = settings.qwen_model
    tools = [_map_tool(tool) for tool in get_mcp_server().list_tools()]
    return [
        {
            "id": "general",
            "name": "智能客服 Agent",
            "description": "电信与电商查询、知识检索及受治理业务办理",
            "status": "idle",
            "tools": tools,
            "model": model or "未配置",
            "icon": "sparkles",
        }
    ]


def list_workspace_files() -> list[dict[str, Any]]:
    root = get_settings().root_dir.resolve()
    return [_file_node(path, root) for path in _visible_children(root)]


def _visible_children(directory: Path) -> list[Path]:
    try:
        children = [
            path
            for path in directory.iterdir()
            if path.name not in EXCLUDED_NAMES and not path.name.startswith(".") and not path.is_symlink()
        ]
    except OSError:
        return []
    return sorted(children, key=lambda path: (not path.is_dir(), path.name.lower()))


def _file_node(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    node: dict[str, Any] = {
        "id": relative,
        "name": path.name,
        "path": relative,
        "type": "folder" if path.is_dir() else "file",
    }
    if path.is_dir():
        node["children"] = [_file_node(child, root) for child in _visible_children(path)]
    else:
        language = LANGUAGES.get(path.suffix.lower())
        if language:
            node["language"] = language
        node["mimeType"] = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return node


def _map_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "category": tool.get("category"),
        "inputSchema": tool.get("inputSchema", tool.get("input_schema", {})),
        "enabled": True,
        "status": "available",
    }
