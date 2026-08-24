from __future__ import annotations

from threading import RLock

from pkg.config.settings import get_settings

_identity_lock = RLock()


def get_local_user_id() -> str:
    """Return the only authoritative identity in local single-user mode."""
    with _identity_lock:
        return get_settings().local_user_id


def set_local_user_id(user_id: str) -> None:
    """Update the authoritative identity after API-level validation."""
    with _identity_lock:
        get_settings().local_user_id = user_id
