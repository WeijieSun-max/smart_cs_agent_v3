from __future__ import annotations

import json
from typing import Any

from domain.customer_service_agent.memory.models import MemoryItem, MemoryStatus, MemoryType
from domain.customer_service_agent.policy.pii import reject_pii
from domain.customer_service_agent.service import memory_service
from pkg.exceptions.exception import StorageUnavailableError


def _repository():
    service = memory_service.get_service_or_none()
    if service is None or not service.repository.available:
        raise StorageUnavailableError()
    return service.repository


def list_memories(
    user_id: str,
    *,
    memory_type: MemoryType | None,
    status: MemoryStatus | None,
    cursor: str | None,
    limit: int,
) -> tuple[list[MemoryItem], str | None]:
    return _repository().list_items(
        user_id,
        memory_type=memory_type,
        status=status,
        cursor=cursor,
        limit=limit,
    )


def correct_memory(
    user_id: str,
    memory_id: str,
    *,
    content: str,
    structured_data: dict[str, Any] | None,
    reason: str,
) -> MemoryItem | None:
    reject_pii(content)
    if structured_data:
        reject_pii(json.dumps(structured_data, ensure_ascii=False, sort_keys=True))
    return _repository().correct_item(
        user_id,
        memory_id,
        content=content,
        structured_data=structured_data,
        reason=reason,
        actor="user",
    )


def forget_memory(user_id: str, memory_id: str) -> bool:
    return _repository().hard_delete_item(
        user_id,
        memory_id,
        reason="user_forget",
        actor="user",
    )


def purge_memories(user_id: str, memory_type: MemoryType | None, reason: str) -> int:
    return _repository().purge_items(
        user_id,
        memory_type,
        reason=reason,
        actor="user",
    )
