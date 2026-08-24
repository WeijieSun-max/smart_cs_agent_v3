from __future__ import annotations

from typing import Any, Optional

from domain.customer_service_agent.interfaces.i_order_query import IOrderQuery
from pkg.telemetry import traced_dependency


class OrderService:
    def __init__(self, order_query: IOrderQuery):
        self.order_query = order_query

    @traced_dependency("order.query", "tool")
    def query_order(self, order_id: str = "", user_id: str = "") -> dict[str, Any]:
        return self.order_query.query_order(order_id=order_id, user_id=user_id)


instance: Optional[OrderService] = None


def initialize_service(order_query: IOrderQuery) -> None:
    global instance
    instance = OrderService(order_query)


def get_service() -> OrderService:
    if instance is None:
        raise RuntimeError("Order service is not initialized")
    return instance
