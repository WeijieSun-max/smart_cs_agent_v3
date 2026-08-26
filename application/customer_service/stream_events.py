from __future__ import annotations

import json
from typing import Any


def encode_sse(payload: dict[str, Any]) -> str:
    return f"data:{json.dumps(payload, ensure_ascii=False)}\n\n"


def cumulative_response_chunks(content: str, max_chars: int = 80) -> list[str]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    return [content[:end] for end in range(max_chars, len(content), max_chars)] + ([content] if content else [])


async def generate_replay_events(turn, content: str):
    yield encode_sse({"type": "meta", "turn_id": turn.turn_id, "trace_id": turn.trace_id, "replayed": True})
    yield encode_sse({"type": "answer", "content": content})
    yield encode_sse({"type": "terminal", "status": "completed", "turn_id": turn.turn_id})
    yield "data:[DONE]\n\n"
