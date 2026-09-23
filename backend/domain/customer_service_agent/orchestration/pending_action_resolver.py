"""把 Supervisor 已分类的确认/拒绝决定映射到治理动作状态机。"""

from __future__ import annotations

import asyncio
import unicodedata
from typing import Any, Literal

from domain.action_governance import get_action_service
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.identity import RequestIdentityContext

_EXPLICIT_ACTION_COMMANDS: dict[str, Literal["confirm_action", "reject_action"]] = {
    "确认": "confirm_action",
    "确认执行": "confirm_action",
    "取消": "reject_action",
    "取消操作": "reject_action",
}


def explicit_pending_action_decision(
    query: str,
) -> Literal["confirm_action", "reject_action"] | None:
    """只识别待确认状态下无歧义的规范命令，不解释一般自然语言。

    NFKC 仅统一全角字符，末尾标点不改变授权含义。包含额外业务文字的
    输入不会命中，仍交给 Supervisor 判断，避免扩大执行权限。
    """

    normalized = unicodedata.normalize("NFKC", query).strip().rstrip("。.!！")
    return _EXPLICIT_ACTION_COMMANDS.get(normalized)

async def resolve_pending_action(
    state: ChatState,
    identity: RequestIdentityContext,
    decision: Literal["confirm_action", "reject_action"] | None = None,
) -> dict[str, Any] | None:
    """对活跃治理动作应用已经分类的决定。

    规范的“确认/取消”命令可由治理状态机确定性分类；其他自然语言仍由
    Supervisor 分类。本层只验证动作是否存在并执行状态迁移。
    `decision=None` 时只生成确认提示。
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
            f"已取消操作：{rejected.impact_summary.rstrip('。')}。",
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
    """构造合并回父图的最小状态增量。"""

    return {
        "intent": intent,
        "current_agent": "action_governance",
        "sub_results": {**state.get("sub_results", {}), "supervisor": text},
        "task_results": task_results or {},
        "node_logs": [f"Pending action resolved: {intent}"],
    }


def _action_result_text(status: str, summary: str, receipt: dict[str, Any] | None) -> str:
    """把治理终态转换为用户可见文本，并明确区分成功与结果未知。"""

    if status == "succeeded":
        receipt_summary = (receipt or {}).get("summary")
        if (receipt or {}).get("resource_type") == "address" and isinstance(receipt_summary, dict):
            address = str(receipt_summary.get("full_address") or "新收货地址")
            contact = str(receipt_summary.get("recipient") or "")
            phone = str(receipt_summary.get("phone") or "")
            default_text = "并已设为默认地址" if receipt_summary.get("is_default") else ""
            contact_text = f"，收件人：{contact}，联系电话：{phone}" if contact and phone else ""
            return f"操作已完成：已创建{address}{contact_text}{default_text}。执行回执已生成，请勿重复提交。"
        return f"操作已完成：{summary.rstrip('。')}。执行回执已生成，请勿重复提交。"
    if status == "indeterminate":
        return "操作结果暂时未知，请勿重复提交；系统将按同一幂等键对账。"
    return f"操作未完成（{status}），数据库未被宣称为成功。"
