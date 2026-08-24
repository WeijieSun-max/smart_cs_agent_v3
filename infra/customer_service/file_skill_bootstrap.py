from __future__ import annotations

from typing import Any

from domain.customer_service_agent.file_skills import get_catalog, initialize_catalog
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from pkg.config.settings import get_settings

_health: dict[str, Any] = {"status": "not_initialized", "count": 0}


def initialize_file_skills() -> None:
    global _health
    settings = get_settings()
    if not settings.skill_files_enabled:
        _health = {"status": "disabled", "count": 0}
        return
    catalog = initialize_catalog(settings.root_dir / settings.skill_root, get_mcp_server())
    summaries = catalog.summaries()
    _health = {"status": "ready", "count": len(summaries), "skills": summaries}


def file_skill_health_status() -> dict[str, Any]:
    return dict(_health)
