from __future__ import annotations

import json
from typing import Any


def encode_sse(payload: dict[str, Any]) -> str:
    return f"data:{json.dumps(payload, ensure_ascii=False)}\n\n"


async def generate_replay_events(turn, content: str):
    yield encode_sse({"type": "meta", "turn_id": turn.turn_id, "trace_id": turn.trace_id, "replayed": True})
    yield encode_sse({"type": "answer", "content": content})
    yield encode_sse({"type": "terminal", "status": "completed", "turn_id": turn.turn_id})
    yield "data:[DONE]\n\n"
