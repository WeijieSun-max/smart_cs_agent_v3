from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ActionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
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
    arguments_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    impact_summary: str
    status: Literal["awaiting_confirmation", "executing", "succeeded", "rejected", "expired", "failed", "indeterminate"]
    idempotency_key: str
    resource_version: int | None = None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    executed_at: datetime | None = None
    receipt: dict[str, Any] | None = None
    error_code: str | None = None
