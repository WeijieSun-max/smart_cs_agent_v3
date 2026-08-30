"""治理动作在提议、确认和执行阶段之间传递的不可变数据模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ActionEnvelope(BaseModel):
    """一次待确认业务写操作的完整快照。

    `arguments_digest` 将调用参数、工具版本和 Skill 版本绑定在一起，防止用户
    确认后参数被替换。`resource_version` 用于乐观并发控制；`idempotency_key`
    则保证超时重试不会重复产生业务副作用。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # 协议与调用方身份。身份字段由可信请求上下文注入，不接受模型生成值。
    schema_version: str = "1.0"
    action_id: str
    user_id: str
    session_id: str
    turn_id: str
    tool_name: str
    tool_version: str
    skill_name: str | None = None
    skill_version: str | None = None
    arguments: dict[str, Any]
    arguments_digest: str = Field(pattern=r"^[0-9a-f]{64}$")    # 将调用参数、工具版本和 Skill 版本绑定在一起
    impact_summary: str

    # 生命周期及幂等控制。indeterminate 表示调用超时后副作用是否发生仍未知。
    status: Literal["awaiting_confirmation", "executing", "succeeded", "rejected", "expired", "failed", "indeterminate"]
    idempotency_key: str
    resource_version: int | None = None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    executed_at: datetime | None = None
    receipt: dict[str, Any] | None = None
    error_code: str | None = None
