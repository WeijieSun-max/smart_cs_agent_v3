from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    user_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    session_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    title: str = Field(default="新会话", max_length=120)
    agent_id: str = Field(default="general", min_length=1, max_length=64)


class SessionUpdateRequest(BaseModel):
    user_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    title: str | None = Field(default=None, max_length=120)
    favorite: bool | None = None


class StopAgentRequest(BaseModel):
    user_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


class CurrentUserUpdateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
