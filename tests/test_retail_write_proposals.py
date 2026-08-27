import asyncio
from types import SimpleNamespace

import pytest

from domain.action_governance import GovernedActionService, initialize_action_service
from domain.business.service import initialize_service
from domain.business.store import InMemoryBusinessStore
from domain.customer_service_agent.orchestration import retail_write_proposals, write_proposal
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from domain.shared.identity import RequestIdentityContext


ORDER = {
    "order_id": "order-1",
    "order_no": "NO-1",
    "user_id": "user-1",
    "status": "pending",
    "version": 3,
    "grand_total": "500.00",
    "currency": "CNY",
    "items": [
        {
            "order_item_id": "item-1",
            "order_id": "order-1",
            "name_snapshot": "移联 X1 手机",
            "sku_snapshot": "PHONE-X1",
            "quantity": 2,
            "returned_qty": 0,
            "exchanged_qty": 0,
        }
    ],
}


class Actions:
    def __init__(self, order=None) -> None:
        self.order = dict(order or ORDER)
        self.proposals = []

    async def execute_read(self, tool_name, arguments, identity):
        assert identity.user_id == "user-1"
        if tool_name == "retail_get_order_detail":
            assert arguments == {"order_id": "order-1"}
            return self.order
        if tool_name == "retail_list_addresses":
            return [{
                "address_id": "address-1",
                "label": "家",
                "province": "北京",
                "city": "北京",
                "district": "海淀",
                "version": 4,
            }]
        if tool_name == "retail_list_payment_methods":
            return [{"payment_method_id": "payment-1", "type": "bank_card", "last4": "1234", "version": 2}]
        raise AssertionError((tool_name, arguments))

    def propose_write(self, tool_name, arguments, identity, *, impact_summary, skill=None):
        del skill
        assert identity.user_id == "user-1"
        self.proposals.append((tool_name, arguments, impact_summary))
        return SimpleNamespace(impact_summary=impact_summary)


def _identity() -> RequestIdentityContext:
    return RequestIdentityContext(user_id="user-1", session_id="session-1", turn_id="turn-1")


def _prepare(monkeypatch, capability: str, query: str, entities: dict[str, str], *, order=None):
    actions = Actions(order)
    monkeypatch.setattr(retail_write_proposals, "get_action_service", lambda: actions)
    result = asyncio.run(write_proposal.prepare_write_proposal(
        query,
        capability,
        _identity(),
        resolved_entities=entities,
    ))
    return result, actions


@pytest.mark.parametrize(
    ("capability", "query", "entities", "tool_name", "expected"),
    [
        (
            "update_order_address",
            "修改订单地址",
            {"order_id": "order-1", "address_id": "address-1"},
            "retail_update_order_address",
            {"order_id": "order-1", "expected_version": 3, "address_id": "address-1"},
        ),
        (
            "update_order_payment",
            "修改订单支付方式",
            {"order_id": "order-1", "payment_method_id": "payment-1"},
            "retail_update_order_payment",
            {"order_id": "order-1", "expected_version": 3, "payment_method_id": "payment-1"},
        ),
        (
            "update_order_items",
            "variant_id:variant-1 quantity:2",
            {"order_id": "order-1", "variant_id": "variant-1", "quantity": 2},
            "retail_update_order_items",
            {"order_id": "order-1", "expected_version": 3, "items": [{"variant_id": "variant-1", "quantity": 2}]},
        ),
    ],
)
def test_pending_order_changes_use_database_version(monkeypatch, capability, query, entities, tool_name, expected) -> None:
    result, actions = _prepare(monkeypatch, capability, query, entities)

    assert result.startswith("待确认：")
    assert actions.proposals[0][0] == tool_name
    assert actions.proposals[0][1] == expected


@pytest.mark.parametrize(
    ("capability", "tool_name"),
    [
        ("request_return", "retail_request_return"),
        ("request_exchange", "retail_request_exchange"),
    ],
)
def test_return_and_exchange_select_owned_order_item(monkeypatch, capability, tool_name) -> None:
    delivered = {**ORDER, "status": "delivered"}
    result, actions = _prepare(
        monkeypatch,
        capability,
        "处理手机 quantity:1",
        {"order_id": "order-1", "product_query": "手机", "reason": "不合适", "quantity": 1},
        order=delivered,
    )

    assert result.startswith("待确认：")
    assert actions.proposals[0][0] == tool_name
    assert actions.proposals[0][1] == {
        "order_id": "order-1",
        "expected_version": 3,
        "items": [{"order_item_id": "item-1", "quantity": 1}],
        "reason": "不合适",
    }


def test_price_adjustment_requires_amount_and_method(monkeypatch) -> None:
    delivered = {**ORDER, "status": "delivered"}
    result, actions = _prepare(
        monkeypatch,
        "price_adjustment_refund",
        "申请退差价 20 元，原路退回",
        {"order_id": "order-1", "amount": 20, "refund_method": "original"},
        order=delivered,
    )

    assert result.startswith("待确认：")
    assert actions.proposals[0][0] == "retail_price_adjustment_refund"
    assert actions.proposals[0][1]["amount"] == 20.0
    assert actions.proposals[0][1]["method"] == "original"


def test_default_address_uses_latest_owned_address_version(monkeypatch) -> None:
    result, actions = _prepare(
        monkeypatch,
        "default_address",
        "设为默认地址",
        {"address_id": "address-1"},
    )

    assert result.startswith("待确认：")
    assert actions.proposals[0][0] == "retail_set_default_address"
    assert actions.proposals[0][1] == {"address_id": "address-1", "expected_version": 4}


def test_ineligible_order_is_not_proposed(monkeypatch) -> None:
    processed = {**ORDER, "status": "processed"}
    result, actions = _prepare(
        monkeypatch,
        "update_order_address",
        "修改订单地址",
        {"order_id": "order-1", "address_id": "address-1"},
        order=processed,
    )

    assert "不能修改收货地址" in result
    assert actions.proposals == []


def test_return_proposal_executes_only_after_governed_confirmation() -> None:
    store = InMemoryBusinessStore({
        "users": [{"user_id": "user-1", "status": "active"}],
        "orders": [{**ORDER, "status": "delivered"}],
        "order_items": ORDER["items"],
    })
    business = initialize_service(store)
    actions = GovernedActionService(get_mcp_server(), business)
    initialize_action_service(actions)
    direct = asyncio.run(get_mcp_server().call_tool(
        "retail_get_order_detail",
        {"order_id": "order-1"},
        trusted_context={"user_id": "user-1"},
    ))
    assert direct.success, (direct.error_type, direct.error_code, direct.error)

    proposal = asyncio.run(write_proposal.prepare_write_proposal(
        "申请手机退货 quantity:1",
        "request_return",
        _identity(),
        resolved_entities={
            "order_id": "order-1",
            "product_query": "手机",
            "reason": "不合适",
            "quantity": 1,
        },
    ))

    assert proposal.startswith("待确认：")
    assert store.get_owned("orders", "order-1", "user-1")["status"] == "delivered"
    completed = asyncio.run(actions.confirm(_identity()))
    assert completed.status == "succeeded"
    assert store.get_owned("orders", "order-1", "user-1")["status"] == "return_requested"
