"""长期记忆、会话摘要、模型上下文和向量索引的严格领域模型。"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictMemoryModel(BaseModel):
    """禁止额外字段并统一清理字符串空白的记忆模型基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MemoryType(StrEnum):
    """记忆语义类型：经历、偏好、稳定事实或未完成任务。"""

    EPISODE = "episode"
    PREFERENCE = "preference"
    FACT = "fact"
    TASK = "task"


class MemoryStatus(StrEnum):
    """记忆从候选到活跃、被取代、完成或过期的生命周期。"""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class MemoryOutboxEventType(StrEnum):
    """权威事务提交后需要异步执行的记忆副作用类型。"""

    MEMORY_EXTRACT = "memory_extract"
    SUMMARY_UPDATE = "summary_update"
    INDEX_UPSERT = "index_upsert"
    INDEX_DELETE = "index_delete"


class MemoryOutboxStatus(StrEnum):
    """outbox 事件的租约消费和重试状态。"""

    PENDING = "pending"
    PROCESSING = "processing"
    RETRY = "retry"
    COMPLETED = "completed"
    DEAD = "dead"


class MemoryCandidate(StrictMemoryModel):
    """模型从对话中抽取、尚未经过策略接受的候选记忆。"""

    memory_type: MemoryType
    memory_key: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    structured_data: dict[str, Any] = Field(default_factory=dict)


class MemoryItem(StrictMemoryModel):
    """权威仓储中的版本化长期记忆。

    同一语义键的新版本通过 `supersedes_id` 指向旧记录。valid 时间描述业务
    事实有效期，expires_at 描述保留策略期限，二者含义不可混用。
    """

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
        """拒绝无时区时间，避免跨服务比较时产生隐式本地时区。"""

        if value is not None and value.tzinfo is None:
            raise ValueError("memory timestamps must be timezone-aware")
        return value


class MemorySource(StrictMemoryModel):
    """记忆与原始会话轮次、人工纠正或业务事件之间的可追溯关系。"""

    memory_id: uuid.UUID
    session_id: str = Field(min_length=1, max_length=128)
    turn_id: str = Field(min_length=1, max_length=64)
    message_id: int | None = Field(default=None, ge=1)
    source_kind: Literal["conversation", "user_correction", "business_event"] = "conversation"
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """保证来源审计时间具有明确时区。"""

        if value.tzinfo is None:
            raise ValueError("source timestamp must be timezone-aware")
        return value


class SessionSummary(StrictMemoryModel):
    """覆盖到指定消息 ID 的不可变、版本化会话摘要。"""

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
        """保证摘要版本时间具有明确时区。"""

        if value.tzinfo is None:
            raise ValueError("summary timestamp must be timezone-aware")
        return value


class ConversationMemoryMessage(StrictMemoryModel):
    """进入记忆编排器的最小用户/助手消息结构。"""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)
    timestamp: datetime | None = None

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        """保证可选消息时间具有明确时区。"""

        if value is not None and value.tzinfo is None:
            raise ValueError("message timestamp must be timezone-aware")
        return value


class MemoryReference(StrictMemoryModel):
    """携带存储 ID 的内部召回结果及最终综合分数。"""

    memory_id: uuid.UUID
    memory_type: MemoryType
    content: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    updated_at: datetime

    @field_validator("updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """保证召回结果的新近度计算使用有时区时间。"""

        if value.tzinfo is None:
            raise ValueError("reference timestamp must be timezone-aware")
        return value


class ConversationContextMemory(StrictMemoryModel):
    """移除存储标识后的 LLM 可见记忆引用。"""

    memory_type: MemoryType
    content: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    updated_at: datetime

    @field_validator("updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """保证 LLM 上下文记忆时间具有明确时区。"""

        if value.tzinfo is None:
            raise ValueError("context memory timestamp must be timezone-aware")
        return value


class ConversationContext(StrictMemoryModel):
    """单轮提供给 LLM 的结构化、不可信参考数据。

    历史摘要和记忆可能陈旧或含提示注入，只能用于指代消解；授权和当前业务
    状态必须由本轮可信工具重新验证。
    """

    summary: str = Field(default="", max_length=12_000)
    recent_messages: list[ConversationMemoryMessage] = Field(default_factory=list, max_length=100)
    memories: list[ConversationContextMemory] = Field(default_factory=list, max_length=200)


class MemoryPacket(StrictMemoryModel):
    """在总 token 预算内组装的摘要、近期对话和长期记忆。"""

    session_summary: str = Field(default="", max_length=12_000)
    recent_messages: list[ConversationMemoryMessage] = Field(default_factory=list, max_length=100)
    episodes: list[MemoryReference] = Field(default_factory=list, max_length=100)
    semantic_memories: list[MemoryReference] = Field(default_factory=list, max_length=100)
    token_count: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=1800, ge=1, le=32_000)

    @model_validator(mode="after")
    def validate_budget(self) -> "MemoryPacket":
        """把预算限制固化为模型不变量，阻止超额载荷进入提示。"""

        if self.token_count > self.max_tokens:
            raise ValueError("memory packet exceeds its token budget")
        return self


class MemoryOutboxEvent(StrictMemoryModel):
    """记忆事务提交后由 worker 租约处理的持久化事件。"""

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
        """保证租约、重试和处理时间可以安全比较。"""

        if value is not None and value.tzinfo is None:
            raise ValueError("outbox timestamp must be timezone-aware")
        return value


class VectorRecord(StrictMemoryModel):
    """写入可重建向量索引的记忆投影和过滤元数据。"""

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
    """向量索引返回的记忆 ID 与归一化相似度。"""

    memory_id: uuid.UUID
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
