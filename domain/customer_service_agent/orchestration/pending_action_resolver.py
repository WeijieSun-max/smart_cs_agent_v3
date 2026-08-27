from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from domain.action_governance import get_action_service
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.identity import RequestIdentityContext

async def resolve_pending_action(
    state: ChatState,
    identity: RequestIdentityContext,
    decision: Literal["confirm_action", "reject_action"] | None = None,
) -> dict[str, Any] | None:
    """Apply an already-classified LLM decision to the active governed action.

    This function deliberately does not interpret user text. The supervisor is
    the only intent classifier; this layer only enforces the action state machine.
    """
    try:
        actions = get_action_service()
    except RuntimeError:
        return None
    active = await asyncio.to_thread(actions.get_active, identity)
    if active is None:
        return None
    if decision == "confirm_action":
        completed = await actions.confirm(identity)
        return _build_result(
            state,
            "action_confirmation",
            _action_result_text(completed.status, completed.impact_summary, completed.receipt),
            task_results={"action": completed.model_dump(mode="json")},
        )
    if decision == "reject_action":
        rejected = await asyncio.to_thread(actions.reject, identity)
        return _build_result(
            state,
            "action_rejection",
            f"已取消操作：{rejected.impact_summary}。",
            task_results={"action": rejected.model_dump(mode="json")},
        )
    return _build_result(
        state,
        "action_pending",
        f"当前有待确认操作：{active.impact_summary}\n请仅回复“确认”执行，或回复“取消”。原参数不会因确认消息而改变。",
    )


def _build_result(
    state: ChatState,
    intent: str,
    text: str,
    *,
    task_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "intent": intent,
        "current_agent": "action_governance",
        "sub_results": {**state.get("sub_results", {}), "supervisor": text},
        "task_results": task_results or {},
        "node_logs": [f"Pending action resolved: {intent}"],
    }


def _action_result_text(status: str, summary: str, receipt: dict[str, Any] | None) -> str:
    if status == "succeeded":
        return f"操作已完成：{summary}。\n执行回执：{json.dumps(receipt or {}, ensure_ascii=False, default=str)}"
    if status == "indeterminate":
        return "操作结果暂时未知，请勿重复提交；系统将按同一幂等键对账。"
    return f"操作未完成（{status}），数据库未被宣称为成功。"
