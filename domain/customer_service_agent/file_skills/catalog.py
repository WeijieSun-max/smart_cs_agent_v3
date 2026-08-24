from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from domain.customer_service_agent.tools.mcp_server import MCPToolServer

from .models import LoadedSkill, SkillIndexEntry, SkillMetadata

_FRONTMATTER = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.DOTALL)
_REFERENCE_LINK = re.compile(r"\[[^\]]+\]\((references/[^)#?]+)(?:#[^)]+)?\)")


class FileSkillCatalog:
    """Frozen catalog whose only business source is ``**/SKILL.md``."""

    def __init__(self, root: Path, tool_server: MCPToolServer):
        self.root = root.resolve()
        self._tool_server = tool_server
        self._entries: dict[tuple[str, str], SkillIndexEntry] = {}
        self._active_by_name: dict[str, SkillIndexEntry] = {}
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def scan_and_freeze(self) -> None:
        if self._frozen:
            raise RuntimeError("skill catalog is already frozen")
        if not self.root.is_dir():
            raise ValueError(f"skill root does not exist: {self.root}")
        paths = sorted(self.root.rglob("SKILL.md"))
        if not paths:
            raise ValueError("no SKILL.md files found")
        for path in paths:
            raw = path.read_text(encoding="utf-8")
            match = _FRONTMATTER.match(raw)
            if match is None:
                raise ValueError(f"missing YAML frontmatter: {path}")
            parsed = yaml.safe_load(match.group(1))
            if not isinstance(parsed, dict):
                raise ValueError(f"invalid YAML frontmatter: {path}")
            metadata = SkillMetadata.model_validate(parsed)
            key = (metadata.name, metadata.version)
            if key in self._entries:
                raise ValueError(f"duplicate skill name/version: {metadata.name}@{metadata.version}")
            self._validate_tool_contract(metadata)
            entry = SkillIndexEntry(
                metadata=metadata,
                path=path.resolve(),
                content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            )
            self._entries[key] = entry
            current = self._active_by_name.get(metadata.name)
            if current is None or _semver(metadata.version) > _semver(current.metadata.version):
                self._active_by_name[metadata.name] = entry
        self._frozen = True

    def summaries(self) -> list[dict[str, object]]:
        self._require_frozen()
        return [
            {
                "name": entry.metadata.name,
                "version": entry.metadata.version,
                "description": entry.metadata.description,
                "domain": entry.metadata.domain,
                "capabilities": list(entry.metadata.capabilities),
                "effect": entry.metadata.effect,
                "content_hash": entry.content_hash,
            }
            for entry in sorted(self._active_by_name.values(), key=lambda item: item.metadata.name)
        ]

    def select(self, *, capability: str, agent_type: str) -> SkillIndexEntry | None:
        self._require_frozen()
        matches = [
            entry for entry in self._active_by_name.values()
            if capability in entry.metadata.capabilities
            and agent_type in entry.metadata.allowed_agent_types
        ]
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous skill capability: {capability}")
        return matches[0] if matches else None

    def load(self, name: str, version: str | None = None, *, agent_type: str) -> LoadedSkill:
        self._require_frozen()
        entry = self._active_by_name.get(name) if version is None else self._entries.get((name, version))
        if entry is None:
            raise KeyError(name)
        if agent_type not in entry.metadata.allowed_agent_types:
            raise PermissionError(f"agent type cannot load skill: {agent_type}")
        raw = entry.path.read_text(encoding="utf-8")
        if hashlib.sha256(raw.encode("utf-8")).hexdigest() != entry.content_hash:
            raise RuntimeError("skill changed after catalog freeze")
        match = _FRONTMATTER.match(raw)
        if match is None:
            raise RuntimeError("skill frontmatter disappeared")
        body = raw[match.end():].strip()
        references: dict[str, str] = {}
        skill_dir = entry.path.parent.resolve()
        for relative in sorted(set(_REFERENCE_LINK.findall(body))):
            target = (skill_dir / relative).resolve()
            if skill_dir not in target.parents or not target.is_file():
                raise ValueError(f"invalid skill reference: {relative}")
            references[relative] = target.read_text(encoding="utf-8")
        return LoadedSkill(
            metadata=entry.metadata,
            body=body,
            references=references,
            path=entry.path,
            content_hash=entry.content_hash,
        )

    def _validate_tool_contract(self, metadata: SkillMetadata) -> None:
        for tool_name in metadata.allowed_tools:
            tool = self._tool_server.get_tool(tool_name)
            if tool is None:
                raise ValueError(f"skill references unknown tool: {tool_name}")
            if metadata.effect == "read" and tool.effect != "read":
                raise ValueError(f"read skill cannot allow write tool: {tool_name}")
            if metadata.allowed_agent_types and tool.allowed_agent_types:
                if not set(metadata.allowed_agent_types).intersection(tool.allowed_agent_types):
                    raise ValueError(f"skill/tool agent contract mismatch: {tool_name}")

    def _require_frozen(self) -> None:
        if not self._frozen:
            raise RuntimeError("skill catalog is not frozen")


def _semver(value: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in value.split("."))  # type: ignore[return-value]


_catalog: FileSkillCatalog | None = None


def initialize_catalog(root: Path, tool_server: MCPToolServer) -> FileSkillCatalog:
    global _catalog
    catalog = FileSkillCatalog(root, tool_server)
    catalog.scan_and_freeze()
    _catalog = catalog
    return catalog


def get_catalog() -> FileSkillCatalog:
    if _catalog is None:
        raise RuntimeError("file skill catalog is not initialized")
    return _catalog
