from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RequestIdentityContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "1.0"
    tenant_id: str = "default"
    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    session_id: str = Field(min_length=1, max_length=128)
    turn_id: str = Field(min_length=1, max_length=64)
    request_id: str | None = None
    identity_source: Literal["request_body", "trusted_proxy_header"] = "request_body"
    auth_strength: Literal["unverified_frontend", "proxy_verified"] = "unverified_frontend"
