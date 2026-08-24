from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from pkg.config.settings import get_settings


class _ChatRequestBase(BaseModel):
    message: str = Field(min_length=1)
    user_id: str | None = Field(default=None, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    request_id: str | None = Field(default=None, min_length=16, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        if len(value) > get_settings().chat_message_max_length:
            raise ValueError("message exceeds configured maximum length")
        return value


class ChatRequest(_ChatRequestBase):
    pass


class ChatStreamRequest(_ChatRequestBase):
    pass


class ChatResponse(BaseModel):
    response: str
    session_id: str
    turn_id: str
    trace_id: str
    intent: str
    compliance_passed: bool


class ToolCallRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    arguments: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    helpful: bool
    reason: Literal["incorrect", "irrelevant", "incomplete", "too_slow", "unsafe"] | None = None


class FeedbackResponse(BaseModel):
    accepted: bool
