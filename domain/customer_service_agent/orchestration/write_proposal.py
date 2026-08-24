from __future__ import annotations

import asyncio
import re

from domain.action_governance import get_action_service
from domain.customer_service_agent.orchestration.planner import extract_entities
from domain.customer_service_agent.orchestration.router import WRITE_CAPABILITIES
from domain.shared.identity import RequestIdentityContext


async def prepare_write_proposal(
    query: str,
    capability: str,
    identity: RequestIdentityContext,
) -> str | None:
    if capability not in WRITE_CAPABILITIES:
        return None
    entities = extract_entities(query)
    if capability == "plan_change":
        if not entities.get("line_id") or not entities.get("plan_id"):
            return "请提供目标 line_id 和 plan_id，我会先展示月租影响，再生成待确认操作。"
        actions = get_action_service()
        quote = await actions.execute_read(
            "telecom_quote_plan_change",
            {"line_id": entities["line_id"], "plan_id": entities["plan_id"]},
            identity,
        )
        arguments = {
            "line_id": quote["line_id"],
            "plan_id": quote["plan_id"],
            "expected_version": quote["expected_version"],
        }
        impact = (
            f"将线路 {quote['line_id']} 从 {quote['current_plan']['name']}"
            f"（{quote['current_plan']['monthly_price']}）变更为 "
            f"{quote['target_plan']['name']}（{quote['target_plan']['monthly_price']}）"
        )
        action = await asyncio.to_thread(
            actions.propose_write,
            "telecom_change_plan",
            arguments,
            identity,
            impact_summary=impact,
        )
        return _confirmation_text(action.impact_summary)
    if capability == "data_refuel":
        amount_match = re.search(r"(\d+)\s*(GB|G)", query, re.I)
        if not entities.get("line_id") or not amount_match:
            return "请提供 line_id 和要补充的 GB 数量。"
        actions = get_action_service()
        amount = int(amount_match.group(1)) * 1024
        quote = await actions.execute_read(
            "telecom_quote_refuel",
            {"line_id": entities["line_id"], "amount_mb": amount},
            identity,
        )
        impact = (
            f"为线路 {quote['line_id']} 补充 {amount // 1024}GB 流量，"
            f"费用 {quote['quoted_price']} {quote['currency']}"
        )
        action = await asyncio.to_thread(
            actions.propose_write,
            "telecom_refuel_data",
            quote,
            identity,
            impact_summary=impact,
        )
        return _confirmation_text(action.impact_summary)
    if capability == "roaming":
        if not entities.get("line_id"):
            return "请提供要办理漫游的 line_id。"
        actions = get_action_service()
        current = await actions.execute_read(
            "telecom_get_current_plan",
            {"line_id": entities["line_id"]},
            identity,
        )
        enabled = not bool(re.search(r"关闭|停用|取消", query))
        arguments = {
            "line_id": entities["line_id"],
            "enabled": enabled,
            "expected_version": current["line_version"],
        }
        impact = f"{'开启' if enabled else '关闭'}线路 {entities['line_id']} 的漫游功能"
        action = await asyncio.to_thread(
            actions.propose_write,
            "telecom_set_roaming",
            arguments,
            identity,
            impact_summary=impact,
        )
        return _confirmation_text(action.impact_summary)
    if capability == "cancel_order":
        if not entities.get("order_id"):
            return "请提供要取消的 order_id。"
        actions = get_action_service()
        order = await actions.execute_read(
            "retail_get_order",
            {"order_id": entities["order_id"]},
            identity,
        )
        arguments = {
            "order_id": order["order_id"],
            "expected_version": order["version"],
            "reason": "changed_mind",
        }
        impact = (
            f"取消订单 {order.get('order_no', order['order_id'])}，当前总额 "
            f"{order.get('grand_total')} {order.get('currency', 'CNY')}"
        )
        action = await asyncio.to_thread(
            actions.propose_write,
            "retail_cancel_order",
            arguments,
            identity,
            impact_summary=impact,
        )
        return _confirmation_text(action.impact_summary)
    return "该操作需要明确的资源 ID 和变更参数。请在工作台填写后，系统会展示冻结的影响摘要并要求二次确认。"


def _confirmation_text(summary: str) -> str:
    return f"待确认：{summary}\n请回复“确认”执行，或回复“取消”。"
