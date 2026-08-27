import asyncio

from domain.business.service import BusinessService
from domain.business.store import InMemoryBusinessStore
from domain.customer_service_agent.orchestration import resource_resolver
from domain.customer_service_agent.orchestration.models import QueryUnderstandingResult
from domain.shared.identity import RequestIdentityContext


ORDERS = [
    {
        "order_id": "order-2",
        "order_no": "NO-2",
        "user_id": "user-1",
        "status": "delivered",
        "placed_at": "2026-08-24T18:00:00+00:00",
        "grand_total": "3999.00",
        "currency": "CNY",
        "version": 2,
    },
    {
        "order_id": "order-1",
        "order_no": "NO-1",
        "user_id": "user-1",
        "status": "delivered",
        "placed_at": "2026-08-24T09:00:00+00:00",
        "grand_total": "2999.00",
        "currency": "CNY",
        "version": 1,
    },
    {
        "order_id": "other-order",
        "order_no": "OTHER",
        "user_id": "user-2",
        "status": "delivered",
        "placed_at": "2026-08-24T12:00:00+00:00",
        "grand_total": "1.00",
        "currency": "CNY",
        "version": 1,
    },
]
ORDER_ITEMS = [
    {"order_item_id": "item-2", "order_id": "order-2", "name_snapshot": "移联 X2 手机", "sku_snapshot": "PHONE-X2", "quantity": 1},
    {"order_item_id": "item-1", "order_id": "order-1", "name_snapshot": "移联 X1 手机", "sku_snapshot": "PHONE-X1", "quantity": 1},
    {"order_item_id": "other-item", "order_id": "other-order", "name_snapshot": "其他手机", "sku_snapshot": "OTHER", "quantity": 1},
]


class Actions:
    def __init__(self, orders) -> None:
        self.orders = orders
        self.calls = []

    async def execute_read(self, tool_name, arguments, identity):
        self.calls.append((tool_name, arguments, identity.user_id))
        return self.orders


def _understanding(*, ordinal: int | None = None) -> QueryUnderstandingResult:
    entities = {"product_query": "手机"}
    if ordinal:
        entities["ordinal"] = ordinal
    return QueryUnderstandingResult(
        standalone_query="申请退回昨天购买的手机",
        domains=("retail",),
        capabilities=("request_return",),
        entities=entities,
        temporal_range={"start": "2026-08-24", "end": "2026-08-25"},
        ambiguity=True,
        confidence=0.95,
        source="llm",
    )


def _identity() -> RequestIdentityContext:
    return RequestIdentityContext(user_id="user-1", session_id="session-1", turn_id="turn-1")


def test_business_order_search_uses_owned_database_rows() -> None:
    store = InMemoryBusinessStore({
        "users": [
            {"user_id": "user-1", "status": "active"},
            {"user_id": "user-2", "status": "active"},
        ],
        "orders": ORDERS,
        "order_items": ORDER_ITEMS,
    })
    service = BusinessService(store)

    result = service.find_orders(
        "user-1",
        start_date="2026-08-24",
        end_date="2026-08-25",
        product_query="手机",
    )

    assert [item["order_id"] for item in result] == ["order-2", "order-1"]
    assert all(item["user_id"] == "user-1" for item in result)
    assert result[0]["items"][0]["name_snapshot"] == "移联 X2 手机"


def test_multiple_candidates_return_safe_numbered_clarification(monkeypatch) -> None:
    actions = Actions(ORDERS[:2])
    monkeypatch.setattr(resource_resolver, "get_action_service", lambda: actions)

    result = asyncio.run(resource_resolver.resolve_order(_understanding(), _identity()))

    assert result.status == "multiple"
    assert "1." in result.user_fragment
    assert "2." in result.user_fragment
    assert "order-1" not in result.user_fragment
    assert actions.calls == [(
        "retail_find_orders",
        {"start_date": "2026-08-24", "end_date": "2026-08-25", "product_query": "手机"},
        "user-1",
    )]


def test_ordinal_selects_database_candidate(monkeypatch) -> None:
    actions = Actions(ORDERS[:2])
    monkeypatch.setattr(resource_resolver, "get_action_service", lambda: actions)

    result = asyncio.run(resource_resolver.resolve_order(_understanding(ordinal=2), _identity()))

    assert result.status == "resolved"
    assert result.order_id == "order-1"
    assert result.candidates[0]["order_no"] == "NO-1"
