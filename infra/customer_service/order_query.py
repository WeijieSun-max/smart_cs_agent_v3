from __future__ import annotations

from typing import Any

from domain.customer_service_agent.interfaces.i_order_query import IOrderQuery


class DemoOrderQuery(IOrderQuery):
    def query_order(self, order_id: str = "", user_id: str = "") -> dict[str, Any]:
        return {
            "order_id": order_id or "ORD-20260401-001",
            "user_id": user_id or "user_001",
            "status": "shipped",
            "logistics": "已出库，预计2天内送达",
            "amount": 299.0,
            "product": "智能理财产品A",
            "created_at": "2026-04-01T10:00:00",
        }
