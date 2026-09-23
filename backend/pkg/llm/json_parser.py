from __future__ import annotations

import json
from typing import Any


def parse_json_object(content: str) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating surrounding model prose but not invalid JSON."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None
