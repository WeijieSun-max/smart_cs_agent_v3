"""检测不应进入长期记忆或摘要的常见个人敏感信息。"""

from __future__ import annotations

import re

from pkg.exceptions.exception import UnsafeInputError

SENSITIVE_PATTERNS = {
    "phone": r"(?<!\d)1[3-9]\d{9}(?!\d)",
    "id_card": r"(?<!\d)\d{17}[\dXx](?!\d)",
    "bank_card": r"(?<!\d)\d{16,19}(?!\d)",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
}


def detect_pii(content: str) -> list[str]:
    """返回文本命中的敏感信息类别，不返回或记录原始匹配值。"""

    return [name for name, pattern in SENSITIVE_PATTERNS.items() if re.search(pattern, content)]


def reject_pii(content: str) -> None:
    """文本包含任一敏感模式时抛出统一安全输入异常。"""

    if detect_pii(content):
        raise UnsafeInputError()
