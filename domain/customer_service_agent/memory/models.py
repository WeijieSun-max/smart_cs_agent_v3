from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictMemoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MemoryType(StrEnum):
    EPISODE = "episode"
    PREFERENCE = "preference"
    FACT = "fact"
    TASK = "task"


class MemoryStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class MemoryOutboxEventType(StrEnum):
    TURN_COMPLETED = "turn_completed"
    INDEX_UPSERT = "index_upsert"
    INDEX_DELETE = "index_delete"


class MemoryOutboxStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    RETRY = "retry"
    COMPLETED = "completed"
    DEAD = "dead"


class MemoryCandidate(StrictMemoryModel):
    memory_type: MemoryType
    memory_key: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    structured_data: dict[str, Any] = Field(default_factory=dict)


class MemoryItem(StrictMemoryModel):
    memory_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    memory_type: MemoryType
    memory_key: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=4000)
    structured_data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    status: MemoryStatus
    version: int = Field(ge=1)
    supersedes_id: uuid.UUID | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    expires_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("valid_from", "valid_until", "expires_at", "created_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("memory timestamps must be timezone-aware")
        return value


class MemorySource(StrictMemoryModel):
    memory_id: uuid.UUID
    session_id: str = Field(min_length=1, max_length=128)
    turn_id: str = Field(min_length=1, max_length=64)
    message_id: int | None = Field(default=None, ge=1)
    source_kind: Literal["conversation", "user_correction", "business_event"] = "conversation"
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("source timestamp must be timezone-aware")
        return value


class SessionSummary(StrictMemoryModel):
    summary_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    session_id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    summary_text: str = Field(min_length=1, max_length=12_000)
    structured_data: dict[str, Any] = Field(default_factory=dict)
    covers_until_message_id: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("summary timestamp must be timezone-aware")
        return value


class ConversationMemoryMessage(StrictMemoryModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)
    timestamp: datetime | None = None

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("message timestamp must be timezone-aware")
        return value


class MemoryReference(StrictMemoryModel):
    memory_id: uuid.UUID
    memory_type: MemoryType
    content: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    updated_at: datetime

    @field_validator("updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("reference timestamp must be timezone-aware")
        return value


class ConversationContextMemory(StrictMemoryModel):
    """LLM-facing memory reference without storage identifiers."""

    memory_type: MemoryType
    content: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    updated_at: datetime

    @field_validator("updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("context memory timestamp must be timezone-aware")
        return value


class ConversationContext(StrictMemoryModel):
    """Structured, untrusted reference data supplied to an LLM in one user turn."""

    summary: str = Field(default="", max_length=12_000)
    recent_messages: list[ConversationMemoryMessage] = Field(default_factory=list, max_length=100)
    memories: list[ConversationContextMemory] = Field(default_factory=list, max_length=200)


class MemoryPacket(StrictMemoryModel):
    session_summary: str = Field(default="", max_length=12_000)
    recent_messages: list[ConversationMemoryMessage] = Field(default_factory=list, max_length=100)
    episodes: list[MemoryReference] = Field(default_factory=list, max_length=100)
    semantic_memories: list[MemoryReference] = Field(default_factory=list, max_length=100)
    token_count: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=1800, ge=1, le=32_000)

    @model_validator(mode="after")
    def validate_budget(self) -> "MemoryPacket":
        if self.token_count > self.max_tokens:
            raise ValueError("memory packet exceeds its token budget")
        return self


class MemoryOutboxEvent(StrictMemoryModel):
    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_type: MemoryOutboxEventType
    aggregate_id: str = Field(min_length=1, max_length=128)
    payload: dict[str, Any] = Field(default_factory=dict)
    status: MemoryOutboxStatus = MemoryOutboxStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    available_at: datetime
    lease_until: datetime | None = None
    last_error_code: str | None = Field(default=None, max_length=128)
    created_at: datetime
    processed_at: datetime | None = None

    @field_validator("available_at", "lease_until", "created_at", "processed_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("outbox timestamp must be timezone-aware")
        return value


class VectorRecord(StrictMemoryModel):
    memory_id: uuid.UUID
    vector: list[float] = Field(min_length=1)
    user_id: str = Field(min_length=1, max_length=128)
    memory_type: MemoryType
    status: MemoryStatus
    source_session_id: str | None = Field(default=None, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    updated_at: datetime
    expires_at: datetime | None = None
    schema_version: int = Field(default=1, ge=1)


class VectorSearchHit(StrictMemoryModel):
    memory_id: uuid.UUID
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
