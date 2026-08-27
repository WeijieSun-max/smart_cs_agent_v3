from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
from typing import Any

from domain.action_governance import get_action_service
from domain.shared.identity import RequestIdentityContext

_RETAIL_CAPABILITIES = frozenset({
    "update_order_address",
    "update_order_payment",
    "update_order_items",
    "request_return",
    "request_exchange",
    "price_adjustment_refund",
    "default_address",
})


async def prepare_retail_write_proposal(
    query: str,
    capability: str,
    identity: RequestIdentityContext,
    entities: dict[str, Any],
) -> str | None:
    if capability not in _RETAIL_CAPABILITIES:
        return None
    if capability == "default_address":
        return await _default_address(identity, entities)
    order_id = entities.get("order_id")
    if not order_id:
        return "请先从本人订单中选择要办理的订单，或请在工作台填写明确的订单信息。"
    actions = get_action_service()
    order = await actions.execute_read("retail_get_order_detail", {"order_id": order_id}, identity)
    if capability == "update_order_address":
        return await _update_address(actions, order, identity, entities)
    if capability == "update_order_payment":
        return await _update_payment(actions, order, identity, entities)
    if capability == "update_order_items":
        return await _update_items(actions, order, identity, query, entities)
    if capability in {"request_return", "request_exchange"}:
        return await _return_or_exchange(actions, order, identity, query, entities, capability)
    return await _price_adjustment(actions, order, identity, query, entities)


async def _update_address(actions, order: dict[str, Any], identity, entities: dict[str, str]) -> str:
    if order.get("status") != "pending":
        return "该订单已不处于待处理状态，不能修改收货地址。"
    address_id = entities.get("address_id")
    if not address_id:
        return "请从本人有效地址中选择新的 address_id。"
    addresses = await actions.execute_read("retail_list_addresses", {}, identity)
    address = next((item for item in addresses if item.get("address_id") == address_id), None)
    if address is None:
        return "没有找到属于当前用户的有效收货地址。"
    impact = (
        f"将订单 {order.get('order_no', order['order_id'])} 的收货地址修改为"
        f"{address.get('label') or '已选地址'}（{address.get('province', '')}{address.get('city', '')}{address.get('district', '')}）"
    )
    return await _propose(
        actions,
        "retail_update_order_address",
        {"order_id": order["order_id"], "expected_version": order["version"], "address_id": address_id},
        identity,
        impact,
    )


async def _update_payment(actions, order: dict[str, Any], identity, entities: dict[str, str]) -> str:
    if order.get("status") != "pending":
        return "该订单已不处于待处理状态，不能修改支付方式。"
    method_id = entities.get("payment_method_id")
    if not method_id:
        return "请从本人有效支付方式中选择 payment_method_id。"
    methods = await actions.execute_read("retail_list_payment_methods", {}, identity)
    method = next((item for item in methods if item.get("payment_method_id") == method_id), None)
    if method is None:
        return "没有找到属于当前用户的有效支付方式。"
    display = str(method.get("type") or "支付方式")
    if method.get("last4"):
        display += f"（尾号 {method['last4']}）"
    impact = f"将订单 {order.get('order_no', order['order_id'])} 的支付方式修改为 {display}"
    return await _propose(
        actions,
        "retail_update_order_payment",
        {"order_id": order["order_id"], "expected_version": order["version"], "payment_method_id": method_id},
        identity,
        impact,
    )


async def _update_items(actions, order: dict[str, Any], identity, query: str, entities: dict[str, str]) -> str:
    if order.get("status") != "pending":
        return "该订单已不处于待处理状态，不能修改商品。"
    items = _variant_items(query, entities)
    if not items:
        return "请提供要保留的 variant_id 和 quantity，例如“variant_id:V1 quantity:2”。"
    impact = f"将订单 {order.get('order_no', order['order_id'])} 的商品调整为 " + "、".join(
        f"{item['variant_id']}×{item['quantity']}" for item in items
    )
    return await _propose(
        actions,
        "retail_update_order_items",
        {"order_id": order["order_id"], "expected_version": order["version"], "items": items},
        identity,
        impact,
    )


async def _return_or_exchange(
    actions,
    order: dict[str, Any],
    identity,
    query: str,
    entities: dict[str, str],
    capability: str,
) -> str:
    if order.get("status") != "delivered":
        return "只有已送达订单可以发起退货或换货申请。"
    selected = _select_order_item(order.get("items", []), entities)
    if isinstance(selected, str):
        return selected
    available = int(selected.get("quantity", 0)) - int(selected.get("returned_qty", 0)) - int(selected.get("exchanged_qty", 0))
    requested = _quantity(query, entities)
    if requested < 1 or requested > available:
        return f"该商品当前最多可申请 {available} 件，请重新提供数量。"
    operation = "退货" if capability == "request_return" else "换货"
    tool_name = "retail_request_return" if capability == "request_return" else "retail_request_exchange"
    reason = (entities.get("reason") or "用户申请")[:128]
    arguments = {
        "order_id": order["order_id"],
        "expected_version": order["version"],
        "items": [{"order_item_id": selected["order_item_id"], "quantity": requested}],
        "reason": reason,
    }
    impact = (
        f"对订单 {order.get('order_no', order['order_id'])} 中的"
        f"{selected.get('name_snapshot') or '商品'}×{requested} 发起{operation}申请，原因：{reason}"
    )
    return await _propose(actions, tool_name, arguments, identity, impact)


async def _price_adjustment(actions, order: dict[str, Any], identity, query: str, entities: dict[str, str]) -> str:
    if order.get("status") != "delivered":
        return "只有已送达订单可以申请差价退款。"
    amount = _refund_amount(query, entities)
    method = _refund_method(query, entities)
    if amount is None or method is None:
        return "请提供退款金额，并明确选择“原路退回”或“礼品卡”。"
    if amount > Decimal(str(order.get("grand_total") or 0)):
        return "差价退款金额不能超过订单总额。"
    arguments = {
        "order_id": order["order_id"],
        "expected_version": order["version"],
        "amount": float(amount),
        "method": method,
    }
    method_text = "原支付方式" if method == "original" else "礼品卡"
    impact = f"为订单 {order.get('order_no', order['order_id'])} 申请差价退款 {amount} {order.get('currency', 'CNY')}，退款至{method_text}"
    return await _propose(actions, "retail_price_adjustment_refund", arguments, identity, impact)


async def _default_address(identity: RequestIdentityContext, entities: dict[str, str]) -> str:
    address_id = entities.get("address_id")
    if not address_id:
        return "请从本人有效地址中选择要设为默认地址的 address_id。"
    actions = get_action_service()
    addresses = await actions.execute_read("retail_list_addresses", {}, identity)
    address = next((item for item in addresses if item.get("address_id") == address_id), None)
    if address is None:
        return "没有找到属于当前用户的有效收货地址。"
    impact = (
        f"将{address.get('label') or '已选地址'}"
        f"（{address.get('province', '')}{address.get('city', '')}{address.get('district', '')}）设为默认收货地址"
    )
    return await _propose(
        actions,
        "retail_set_default_address",
        {"address_id": address_id, "expected_version": address["version"]},
        identity,
        impact,
    )


async def _propose(actions, tool_name: str, arguments: dict[str, Any], identity, impact: str) -> str:
    action = await asyncio.to_thread(
        actions.propose_write,
        tool_name,
        arguments,
        identity,
        impact_summary=impact,
    )
    return f"待确认：{action.impact_summary}\n请回复“确认”执行，或回复“取消”。"


def _variant_items(query: str, entities: dict[str, Any]) -> list[dict[str, Any]]:
    del query
    raw_items = entities.get("items")
    if isinstance(raw_items, list):
        result = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            variant_id = item.get("variant_id")
            quantity = item.get("quantity")
            if (
                isinstance(variant_id, str)
                and isinstance(quantity, int)
                and not isinstance(quantity, bool)
                and 1 <= quantity <= 99
            ):
                result.append({"variant_id": variant_id, "quantity": quantity})
        return result
    variant_id = entities.get("variant_id")
    quantity = entities.get("quantity", 1)
    if (
        isinstance(variant_id, str)
        and isinstance(quantity, int)
        and not isinstance(quantity, bool)
        and 1 <= quantity <= 99
    ):
        return [{"variant_id": variant_id, "quantity": quantity}]
    return []


def _select_order_item(items: list[dict[str, Any]], entities: dict[str, Any]) -> dict[str, Any] | str:
    selectable = [
        item for item in items
        if int(item.get("quantity", 0)) > int(item.get("returned_qty", 0)) + int(item.get("exchanged_qty", 0))
    ]
    item_id = entities.get("order_item_id")
    if item_id:
        selected = next((item for item in selectable if item.get("order_item_id") == item_id), None)
        return selected or "没有找到属于该订单的可办理商品明细。"
    product_query = (entities.get("product_query") or "").strip().lower()
    if product_query:
        selectable = [
            item for item in selectable
            if product_query in f"{item.get('name_snapshot', '')} {item.get('sku_snapshot', '')}".lower()
        ]
    if len(selectable) == 1:
        return selectable[0]
    if not selectable:
        return "该订单中没有符合描述且可办理的商品。"
    names = "、".join(str(item.get("name_snapshot") or "商品") for item in selectable[:10])
    return f"该订单有多个可办理商品：{names}。请补充商品名称后再试。"


def _quantity(query: str, entities: dict[str, Any]) -> int:
    del query
    value = entities.get("quantity")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 1


def _refund_amount(query: str, entities: dict[str, Any]) -> Decimal | None:
    del query
    value = entities.get("amount")
    raw = value
    if raw is None:
        return None
    try:
        amount = Decimal(raw)
    except InvalidOperation:
        return None
    return amount if Decimal("0") < amount <= Decimal("100000") else None


def _refund_method(query: str, entities: dict[str, Any]) -> str | None:
    del query
    value = entities.get("refund_method")
    if value in {"original", "gift_card"}:
        return value
    return None
