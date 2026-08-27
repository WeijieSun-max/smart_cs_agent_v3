from __future__ import annotations

from infra.business.mysql_business_store import (
    _decode_sensitive_json,
    _encode_sensitive_json,
)
from pkg.security import decrypt_pii, encrypt_pii


def test_business_pii_cipher_round_trip_is_not_plaintext() -> None:
    plaintext = "张伟 18060815554 文艺路9号"

    encrypted = encrypt_pii(plaintext)

    assert plaintext.encode("utf-8") not in encrypted
    assert decrypt_pii(encrypted) == plaintext


def test_governed_action_arguments_are_encrypted_at_rest() -> None:
    arguments = {
        "recipient": "张伟",
        "phone": "18060815554",
        "detail": "文艺路9号南京邮电大学仙林校区东门",
    }

    encoded = _encode_sensitive_json(arguments)

    assert "18060815554" not in encoded
    assert _decode_sensitive_json(encoded) == arguments
