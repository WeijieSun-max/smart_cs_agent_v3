from __future__ import annotations

import re
from typing import Any

from pkg.exceptions.exception import UnsafeInputError

SENSITIVE_PATTERNS = {
    "phone": r"(?<!\d)1[3-9]\d{9}(?!\d)",
    "id_card": r"(?<!\d)\d{17}[\dXx](?!\d)",
    "bank_card": r"(?<!\d)\d{16,19}(?!\d)",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
}


def detect_pii(content: str) -> list[str]:
    return [name for name, pattern in SENSITIVE_PATTERNS.items() if re.search(pattern, content)]


def mask_pii(content: str) -> str:
    masked = content
    for pattern in SENSITIVE_PATTERNS.values():
        def repl(match: re.Match[str]) -> str:
            text = match.group()
            if len(text) <= 4:
                return "****"
            return text[:3] + "*" * (len(text) - 6) + text[-3:]

        masked = re.sub(pattern, repl, masked)
    return masked


def reject_pii(content: str) -> None:
    if detect_pii(content):
        raise UnsafeInputError()


def reject_pii_in_value(value: Any) -> None:
    if isinstance(value, str):
        reject_pii(value)
    elif isinstance(value, dict):
        for item in value.values():
            reject_pii_in_value(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            reject_pii_in_value(item)
