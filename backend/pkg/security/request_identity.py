from __future__ import annotations

import ipaddress

from fastapi import Request

from domain.shared.identity import RequestIdentityContext
from pkg.config.settings import get_settings


def resolve_request_user(request: Request, body_user_id: str | None) -> tuple[str, str, str]:
    settings=get_settings(); source="request_body"; strength="unverified_frontend"; user_id=body_user_id or settings.local_user_id
    if settings.identity_header_enabled:
        address=request.client.host if request.client else ""
        networks=[ipaddress.ip_network(item.strip()) for item in settings.trusted_proxy_networks.split(",") if item.strip()]
        trusted=bool(address and any(ipaddress.ip_address(address) in network for network in networks))
        header=request.headers.get("X-User-Id")
        if trusted and header:
            user_id=header; source="trusted_proxy_header"; strength="proxy_verified"
    RequestIdentityContext(user_id=user_id,session_id="validation",turn_id="validation",identity_source=source,auth_strength=strength)
    return user_id,source,strength
