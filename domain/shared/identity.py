"""从应用请求传入领域执行边界的不可变身份上下文。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RequestIdentityContext(BaseModel):
    """工具授权、资源所有权和审计共同使用的可信调用身份。

    `request_body` 当前只代表前端声明身份，不等同于强认证；只有来自受信代理
    网络并经过上游验证的请求才能标记 `proxy_verified`。模型不得构造此对象。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "1.0"
    tenant_id: str = "default"
    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    session_id: str = Field(min_length=1, max_length=128)
    turn_id: str = Field(min_length=1, max_length=64)
    request_id: str | None = None
    identity_source: Literal["request_body", "trusted_proxy_header"] = "request_body"
    auth_strength: Literal["unverified_frontend", "proxy_verified"] = "unverified_frontend"
