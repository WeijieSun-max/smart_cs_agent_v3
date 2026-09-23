from __future__ import annotations

import base64
import hashlib
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from pkg.config.settings import get_settings

_PREFIX = b"enc:v1:"
_LEGACY_DEMO_PREFIX = b"demo:v1:"


def encrypt_pii(value: str) -> bytes:
    """Encrypt business PII for storage in *_cipher columns."""
    if not isinstance(value, str) or not value:
        raise ValueError("PII value must be a non-empty string")
    return _PREFIX + _fernet().encrypt(value.encode("utf-8"))


def decrypt_pii(value: bytes | bytearray | str | None) -> str | None:
    """Decrypt current values and tolerate explicit legacy/plain test fixtures."""
    if value is None:
        return None
    raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    if raw.startswith(_LEGACY_DEMO_PREFIX):
        # Historical demo hashes are intentionally non-reversible.
        return None
    if raw.startswith(_PREFIX):
        try:
            return _fernet().decrypt(raw[len(_PREFIX):]).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError):
            return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    settings = get_settings()
    secret = settings.pii_encryption_key.strip()
    if not secret:
        if not settings.debug:
            raise RuntimeError("PII_ENCRYPTION_KEY is required outside debug mode")
        secret = (
            f"development-only|{settings.app_name}|"
            f"{settings.db_name}|{settings.local_user_id}"
        )
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    return Fernet(key)

