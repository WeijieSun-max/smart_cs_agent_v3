from __future__ import annotations

import re
from typing import Any

from domain.action_governance import get_action_service
from domain.customer_service_agent.orchestration.models import OrderResolution, QueryUnderstandingResult
from domain.shared.identity import RequestIdentityContext

_ORDER_CAPABILITIES = frozenset({
    "order_query",
    "cancel_order",
    "update_order_address",
    "update_order_payment",
    "update_order_items",
    "request_return",
    "request_exchange",
    "price_adjustment_refund",
})
_STATUS_NAMES = {
    "pending": "待处理",
    "processed": "处理中",
    "delivered": "已送达",
    "cancelled": "已取消",
    "return_requested": "已申请退货",
    "exchange_requested": "已申请换货",
}
_CHINESE_ORDINALS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def needs_order_resolution(capability: str, understanding: QueryUnderstandingResult) -> bool:
    return capability in _ORDER_CAPABILITIES and not understanding.entities.get("order_id")


async def resolve_order(
    understanding: QueryUnderstandingResult,
    identity: RequestIdentityContext,
) -> OrderResolution:
    explicit_order_id = understanding.entities.get("order_id")
    if explicit_order_id:
        return OrderResolution(status="resolved", order_id=explicit_order_id)
    temporal = understanding.temporal_range or {}
    arguments = {
        key: value
        for key, value in {
            "start_date": temporal.get("start"),
            "end_date": temporal.get("end"),
            "product_query": understanding.entities.get("product_query") or "",
            "status": understanding.entities.get("status"),
        }.items()
        if value not in {None, ""}
    }
    orders = await get_action_service().execute_read("retail_find_orders", arguments, identity)
    ordinal = _ordinal(understanding)
    if ordinal is not None:
        if 1 <= ordinal <= len(orders):
            selected = orders[ordinal - 1]
            return OrderResolution(
                status="resolved",
                order_id=str(selected["order_id"]),
                candidates=(_public_candidate(selected),),
            )
        return OrderResolution(
            status="not_found",
            user_fragment=f"当前候选订单只有 {len(orders)} 个，无法选择第 {ordinal} 个。请重新描述要处理的订单。",
        )
    if not orders:
        return OrderResolution(
            status="not_found",
            user_fragment="没有找到符合时间、商品和状态条件的本人订单。请调整日期或商品描述后重试。",
        )
    if len(orders) == 1:
        return OrderResolution(
            status="resolved",
            order_id=str(orders[0]["order_id"]),
            candidates=(_public_candidate(orders[0]),),
        )
    candidates = tuple(_public_candidate(order) for order in orders[:5])
    return OrderResolution(
        status="multiple",
        candidates=candidates,
        user_fragment=_candidate_prompt(candidates, len(orders)),
    )


def _ordinal(understanding: QueryUnderstandingResult) -> int | None:
    value = understanding.entities.get("ordinal") or ""
    match = re.search(r"\d+", value)
    if match:
        return int(match.group())
    for character, number in _CHINESE_ORDINALS.items():
        if character in value:
            return number
    match = re.search(r"第\s*(\d+)\s*个", understanding.standalone_query)
    if match:
        return int(match.group(1))
    for character, number in _CHINESE_ORDINALS.items():
        if f"第{character}个" in understanding.standalone_query:
            return number
    return None


def _public_candidate(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_no": order.get("order_no"),
        "status": order.get("status"),
        "placed_at": order.get("placed_at"),
        "grand_total": order.get("grand_total"),
        "currency": order.get("currency", "CNY"),
        "items": [
            {
                "name": item.get("name_snapshot"),
                "quantity": item.get("quantity"),
            }
            for item in order.get("items", [])[:5]
        ],
    }


def _candidate_prompt(candidates: tuple[dict[str, Any], ...], total: int) -> str:
    lines = ["找到多个符合条件的本人订单，请回复要处理的序号："]
    for index, candidate in enumerate(candidates, start=1):
        names = "、".join(
            f"{item.get('name') or '商品'}×{item.get('quantity') or 1}"
            for item in candidate.get("items", [])
        ) or "商品明细不可用"
        placed_at = str(candidate.get("placed_at") or "未知时间").replace("T", " ")[:19]
        status = _STATUS_NAMES.get(str(candidate.get("status")), str(candidate.get("status") or "未知"))
        lines.append(
            f"{index}. {candidate.get('order_no') or '订单'}｜{placed_at}｜{names}｜{status}｜"
            f"{candidate.get('grand_total')} {candidate.get('currency')}"
        )
    if total > len(candidates):
        lines.append(f"仅展示最近 {len(candidates)} 个候选，共找到 {total} 个；可补充更精确的日期或商品名称。")
    return "\n".join(lines)
