from __future__ import annotations

import asyncio
import math
from typing import Any

from langchain_core.callbacks.manager import adispatch_custom_event

_TELEMETRY_TIMEOUT_SECONDS = 0.05
_EVENT_NAMES = frozenset(
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
_SKILLS = frozenset(
    {
        "onboarding_process_guide",
        "onboarding_material_check",
        "onboarding_risk_assessment",
        "onboarding_eligibility_check",
        "none",
    }
)
_TOOLS = frozenset(
    {"knowledge_search", "user_profile", "risk_check", "ticket_create"}
)
_STATUSES = frozenset(
    {
        "candidates",
        "selected",
        "started",
        "success",
        "failed",
        "rejected",
        "call_tool",
        "ask_clarification",
        "request_confirmation",
        "finish",
        "needs_clarification",
        "awaiting_confirmation",
        "indeterminate",
        "budget_exhausted",
        "completed",
    }
)
_ERROR_CODES = frozenset(
    {
        "skill.invalid_decision",
        "skill.tool_failed",
        "skill.write_status_unknown",
        "skill.tool_not_allowed",
        "skill.validation_failed",
        "skill.unauthorized_write",
        "tool.validation",
    }
)
_COUNT_LIMITS = {
    "candidate_count": 4,
    "selector_llm_decisions": 1,
    "runtime_llm_decisions": 3,
    "remaining_llm_decisions": 3,
    "tool_calls": 5,
    "remaining_tool_calls": 5,
}
_skill_event_tasks: set[asyncio.Task[None]] = set()


def _safe_number(value: object, maximum: float) -> float | None:
    if type(value) is int:
        return float(value) if 0 <= value <= maximum else None
    if type(value) is float and math.isfinite(value) and 0 <= value <= maximum:
        return value
    return None


def safe_skill_event_data(data: object) -> dict[str, Any]:
    if type(data) is not dict:
        return {}
    source = data
    safe: dict[str, Any] = {}
    skill_name = source.get("skill_name")
    if type(skill_name) is str and skill_name in _SKILLS:
        safe["skill_name"] = skill_name
    tool_name = source.get("tool_name")
    if type(tool_name) is str and tool_name in _TOOLS:
        safe["tool_name"] = tool_name
    status = source.get("status")
    if type(status) is str and status in _STATUSES:
        safe["status"] = status
    error_code = source.get("error_code")
    if type(error_code) is str and error_code in _ERROR_CODES:
        safe["error_code"] = error_code
    confidence = _safe_number(source.get("confidence"), 1)
    if confidence is not None:
        safe["confidence"] = confidence
    duration = _safe_number(source.get("duration_ms"), 30_000)
    if duration is not None:
        safe["duration_ms"] = duration
    for field, maximum in _COUNT_LIMITS.items():
        value = source.get(field)
        if type(value) is int and 0 <= value <= maximum:
            safe[field] = value
    return safe


async def _dispatch(name: str, data: dict[str, Any]) -> None:
    try:
        async with asyncio.timeout(_TELEMETRY_TIMEOUT_SECONDS):
            await adispatch_custom_event(name, data)
    except BaseException:
        # Callback cancellation belongs to telemetry, not the business operation.
        return


async def emit_skill_event(
    name: object, data: object, *, yield_control: bool = False
) -> None:
    """Schedule bounded telemetry without waiting for callback completion."""
    task: asyncio.Task[None] | None = None
    if type(name) is str and name in _EVENT_NAMES:
        task = asyncio.create_task(_dispatch(name, safe_skill_event_data(data)))
        _skill_event_tasks.add(task)
        task.add_done_callback(_skill_event_tasks.discard)
    current = asyncio.current_task()
    if current is not None and current.cancelling():
        if task is not None:
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        raise asyncio.CancelledError
    if not yield_control:
        return
    try:
        # Terminal emissions yield once, outside the business execution deadline.
        await asyncio.sleep(0)
    except asyncio.CancelledError:
        if task is not None:
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        raise


def pending_skill_event_task_count() -> int:
    return sum(not task.done() for task in tuple(_skill_event_tasks))


__all__ = [
    "emit_skill_event",
    "pending_skill_event_task_count",
    "safe_skill_event_data",
]
