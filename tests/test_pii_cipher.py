from __future__ import annotations

from infra.business.mysql_business_store import (
    MySQLBusinessStore,
    _decode_sensitive_json,
    _encode_sensitive_json,
)
from domain.business.service import BusinessService
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


def test_mysql_address_cipher_is_decrypted_only_at_business_boundary() -> None:
    class Client:
        def execute_query(self, sql, _args=None, fetch_one=False):
            if "FROM cs_users" in sql:
                return True, {"user_id": "u1"}
            assert "FROM cs_user_addresses" in sql
            rows = [{
                "address_id": "A1",
                "user_id": "u1",
                "recipient_cipher": encrypt_pii("张伟"),
                "phone_cipher": encrypt_pii("18060815554"),
                "detail_cipher": encrypt_pii("南京路88号"),
                "province": "河南省",
                "city": "郑州市",
                "district": "二七区",
                "is_default": 1,
                "status": "active",
                "version": 1,
            }]
            return True, rows[0] if fetch_one else rows

    service = BusinessService(MySQLBusinessStore(Client()))

    result = service.get_default_address("u1")

    assert result["status"] == "found"
    assert result["address"]["recipient"] == "张伟"
    assert result["address"]["phone"] == "18060815554"
    assert result["address"]["detail"] == "南京路88号"
    assert not any(key.endswith("_cipher") for key in result["address"])
