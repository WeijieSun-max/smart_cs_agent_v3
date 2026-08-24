from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from domain.customer_service_agent.interfaces.i_user_profile_repository import IUserProfileRepository
from domain.customer_service_agent.memory.models import MemoryItem, MemoryStatus, MemoryType
from pkg.log.logger import get_logger
from pkg.telemetry import normalize_error, record_fallback


logger = get_logger()
PROFILE_MEMORY_TYPES = [MemoryType.PREFERENCE, MemoryType.FACT]
PROFILE_MEMORY_LIMIT = 50


class MySQLMemoryUserProfileRepository(IUserProfileRepository):
    """Derive a bounded profile view from authoritative MySQL memories."""

    def __init__(
        self,
        memory_repository=None,
        *,
        limit: int = PROFILE_MEMORY_LIMIT,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.memory_repository = memory_repository
        self.limit = max(1, min(PROFILE_MEMORY_LIMIT, limit))
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def get_profile(self, user_id: str) -> dict[str, Any]:
        default = _default_profile(user_id)
        repository = self.memory_repository
        if repository is None or not repository.available:
            record_fallback("memory_user_profile")
            return default

        now = self.clock()
        try:
            candidates = repository.list_active_items(
                user_id,
                memory_types=PROFILE_MEMORY_TYPES,
                limit=self.limit,
                now=now,
            )
        except Exception as exc:
            error = normalize_error(exc)
            logger.warning(
                "Memory user profile query failed error_type={} error_code={}",
                error["error_type"],
                error["error_code"],
            )
            record_fallback("memory_user_profile")
            return default

        memories = sorted(
            (item for item in candidates if _valid_profile_memory(item, user_id, now)),
            key=lambda item: (item.updated_at, str(item.memory_id)),
            reverse=True,
        )[: self.limit]
        if not memories:
            return default

        profile = {**default, "profile_source": "memory"}
        level = _latest_string(memories, "customer_level")
        risk_preference = _latest_string(memories, "risk_preference")
        products = _latest_string_list(memories, "preferred_product_types")
        risk_tags = _latest_string_list(memories, "risk_tags")
        contact_channel = _latest_string(memories, "preferred_contact_channel")
        if level is not None:
            profile["level"] = level
        if risk_preference is not None:
            profile["risk_preference"] = risk_preference
        if products is not None:
            profile["products"] = products
        if risk_tags is not None:
            profile["risk_tags"] = risk_tags
        if contact_channel is not None:
            profile["profile_attributes"] = {"preferred_contact_channel": contact_channel}
        profile["profile_memories"] = [
            {
                "memory_type": item.memory_type.value,
                "memory_key": item.memory_key,
                "content": item.content,
                "updated_at": item.updated_at.isoformat(),
            }
            for item in memories
        ]
        return profile


def _default_profile(user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "level": "standard",
        "risk_preference": "balanced",
        "products": [],
        "risk_tags": [],
        "recent_ticket_count": 0,
        "profile_source": "default",
        "profile_attributes": {},
        "profile_memories": [],
    }


def _valid_profile_memory(item: MemoryItem, user_id: str, now: datetime) -> bool:
    return bool(
        item.user_id == user_id
        and item.memory_type in PROFILE_MEMORY_TYPES
        and item.status == MemoryStatus.ACTIVE
        and (item.expires_at is None or item.expires_at > now)
        and (item.valid_from is None or item.valid_from <= now)
        and (item.valid_until is None or item.valid_until > now)
    )


def _latest_string(memories: list[MemoryItem], field: str) -> str | None:
    for item in memories:
        value = item.structured_data.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _latest_string_list(memories: list[MemoryItem], field: str) -> list[str] | None:
    for item in memories:
        value = item.structured_data.get(field)
        if not isinstance(value, list) or not value:
            continue
        if not all(isinstance(entry, str) and entry.strip() for entry in value):
            continue
        return list(dict.fromkeys(entry.strip() for entry in value))
    return None
