from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from domain.customer_service_agent.memory.models import MemoryStatus, MemoryType


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MemoryCorrectionRequest(StrictRequest):
    user_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    content: str = Field(min_length=1, max_length=4000)
    structured_data: dict[str, Any] | None = None
    reason: str = Field(default="user_correction", min_length=1, max_length=255)


class MemoryPurgeRequest(StrictRequest):
    user_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    confirmation: Literal["PURGE"]
    memory_type: MemoryType | None = None
    reason: str = Field(default="user_purge", min_length=1, max_length=255)


class MemoryItemResponse(BaseModel):
    memory_id: str
    memory_type: MemoryType
    memory_key: str
    content: str
    structured_data: dict[str, Any]
    confidence: float
    status: MemoryStatus
    version: int
    supersedes_id: str | None
    expires_at: str | None
    created_at: str
    updated_at: str


class MemoryListResponse(BaseModel):
    items: list[MemoryItemResponse]
    next_cursor: str | None = None


class MemoryPurgeResponse(BaseModel):
    deleted_count: int = Field(ge=0)
