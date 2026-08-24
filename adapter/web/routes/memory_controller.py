from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, Response, status

from adapter.web.schemas.memory import (
    MemoryCorrectionRequest,
    MemoryItemResponse,
    MemoryListResponse,
    MemoryPurgeRequest,
    MemoryPurgeResponse,
)
from application.customer_service import memory_management_service
from domain.customer_service_agent.memory.models import MemoryItem, MemoryStatus, MemoryType
from pkg.security import get_local_user_id


router = APIRouter(prefix="/api/memories")
MemoryId = Annotated[str, Path(pattern=r"^[0-9a-fA-F-]{36}$")]


@router.get("", response_model=MemoryListResponse)
def list_memories(
    memory_type: MemoryType | None = Query(default=None, alias="type"),
    memory_status: MemoryStatus | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=100),
) -> MemoryListResponse:
    items, next_cursor = memory_management_service.list_memories(
        get_local_user_id(),
        memory_type=memory_type,
        status=memory_status,
        cursor=cursor,
        limit=limit,
    )
    return MemoryListResponse(items=[_response(item) for item in items], next_cursor=next_cursor)


@router.post("/purge", response_model=MemoryPurgeResponse)
def purge_memories(request: MemoryPurgeRequest) -> MemoryPurgeResponse:
    deleted = memory_management_service.purge_memories(
        get_local_user_id(),
        request.memory_type,
        request.reason,
    )
    return MemoryPurgeResponse(deleted_count=deleted)


@router.patch("/{memory_id}", response_model=MemoryItemResponse)
def correct_memory(memory_id: MemoryId, request: MemoryCorrectionRequest) -> MemoryItemResponse:
    item = memory_management_service.correct_memory(
        get_local_user_id(),
        memory_id,
        content=request.content,
        structured_data=request.structured_data,
        reason=request.reason,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return _response(item)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def forget_memory(memory_id: MemoryId) -> Response:
    if not memory_management_service.forget_memory(get_local_user_id(), memory_id):
        raise HTTPException(status_code=404, detail="记忆不存在")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _response(item: MemoryItem) -> MemoryItemResponse:
    return MemoryItemResponse(
        memory_id=str(item.memory_id),
        memory_type=item.memory_type,
        memory_key=item.memory_key,
        content=item.content,
        structured_data=item.structured_data,
        confidence=item.confidence,
        status=item.status,
        version=item.version,
        supersedes_id=str(item.supersedes_id) if item.supersedes_id else None,
        expires_at=item.expires_at.isoformat() if item.expires_at else None,
        created_at=item.created_at.isoformat(),
        updated_at=item.updated_at.isoformat(),
    )
