from __future__ import annotations

import json
from typing import Any


def encode_sse(payload: dict[str, Any]) -> str:
    return f"data:{json.dumps(payload, ensure_ascii=False)}\n\n"


def response_delta_chunks(content: str, max_chars: int = 12) -> list[str]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    return [content[start : start + max_chars] for start in range(0, len(content), max_chars)]


async def generate_replay_events(turn, content: str):
    yield encode_sse({"type": "meta", "turn_id": turn.turn_id, "trace_id": turn.trace_id, "replayed": True})
    yield encode_sse({"type": "answer", "content": content})
    yield encode_sse({"type": "terminal", "status": "completed", "turn_id": turn.turn_id})
    yield "data:[DONE]\n\n"
