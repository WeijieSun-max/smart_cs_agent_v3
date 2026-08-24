from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from domain.action_governance import get_action_service
from domain.customer_service_agent.orchestration.state_updates import build_supervisor_result
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.identity import RequestIdentityContext

_CONFIRM = re.compile(r"^(确认|确认执行|同意|是的|yes|confirm)[。！!\s]*$", re.I)
_REJECT = re.compile(r"^(取消|拒绝|不同意|不要|no|cancel)[。！!\s]*$", re.I)


async def resolve_pending_action(
    state: ChatState,
    identity: RequestIdentityContext,
) -> dict[str, Any] | None:
    try:
        actions = get_action_service()
    except RuntimeError:
        return None
    active = await asyncio.to_thread(actions.get_active, identity)
    if active is None:
        return None
    query = (state.get("raw_query") or "").strip()
    if _CONFIRM.match(query):
        completed = await actions.confirm(identity)
        return build_supervisor_result(
            state,
            "action_confirmation",
            _action_result_text(completed.status, completed.impact_summary, completed.receipt),
            task_results={"action": completed.model_dump(mode="json")},
        )
    if _REJECT.match(query):
        rejected = await asyncio.to_thread(actions.reject, identity)
        return build_supervisor_result(
            state,
            "action_rejection",
            f"已取消操作：{rejected.impact_summary}。",
            task_results={"action": rejected.model_dump(mode="json")},
        )
    return build_supervisor_result(
        state,
        "action_pending",
        f"当前有待确认操作：{active.impact_summary}\n请仅回复“确认”执行，或回复“取消”。原参数不会因确认消息而改变。",
    )


def _action_result_text(status: str, summary: str, receipt: dict[str, Any] | None) -> str:
    if status == "succeeded":
        return f"操作已完成：{summary}。\n执行回执：{json.dumps(receipt or {}, ensure_ascii=False, default=str)}"
    if status == "indeterminate":
        return "操作结果暂时未知，请勿重复提交；系统将按同一幂等键对账。"
    return f"操作未完成（{status}），数据库未被宣称为成功。"
