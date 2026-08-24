from __future__ import annotations

import json
from typing import Any

from pkg.telemetry.skill_events import safe_skill_event_data

_SKILL_EVENT_NAMES = frozenset(
    {
        "skill_candidates",
        "skill_selected",
        "skill_llm_decision",
        "skill_tool_start",
        "skill_tool_end",
        "skill_confirmation_required",
        "skill_completed",
    }
)


def encode_sse(payload: dict[str, Any]) -> str:
    return f"data:{json.dumps(payload, ensure_ascii=False)}\n\n"


async def generate_replay_events(turn, content: str):
    yield encode_sse({"type": "meta", "turn_id": turn.turn_id, "trace_id": turn.trace_id, "replayed": True})
    yield encode_sse({"type": "answer", "content": content})
    yield encode_sse({"type": "terminal", "status": "completed", "turn_id": turn.turn_id})
    yield "data:[DONE]\n\n"


def safe_skill_sse_event(source: object) -> dict[str, Any] | None:
    if type(source) is not dict:
        return None
    source_event = source.get("event")
    if type(source_event) is not str or source_event != "on_custom_event":
        return None
    event_name = source.get("name")
    if type(event_name) is not str or event_name not in _SKILL_EVENT_NAMES:
        return None
    data = source.get("data")
    if type(data) is not dict:
        return None
    return {
        "type": "skill_event",
        "event": event_name,
        **safe_skill_event_data(data),
    }
