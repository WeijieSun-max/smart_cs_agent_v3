from domain.customer_service_agent.orchestration.router import WRITE_CAPABILITIES, route_request


def test_router_routes_telecom_read() -> None:
    decision = route_request("查询当前套餐")

    assert decision.domains == ("telecom",)
    assert decision.capabilities == ("current_plan",)
    assert decision.composite is False
    assert decision.risk_level == "low"


def test_router_routes_retail_write() -> None:
    decision = route_request("取消订单 order_id:ord-1")

    assert decision.domains == ("retail",)
    assert decision.capabilities == ("cancel_order",)
    assert decision.risk_level == "medium"
    assert "cancel_order" in WRITE_CAPABILITIES


def test_router_marks_cross_domain_request_composite() -> None:
    decision = route_request("查询当前套餐并取消订单 order_id:ord-1")

    assert decision.domains == ("telecom", "retail")
    assert decision.capabilities == ("current_plan", "cancel_order")
    assert decision.composite is True
    assert decision.risk_level == "medium"


def test_router_falls_back_for_unknown_request() -> None:
    decision = route_request("你好")

    assert decision.domains == ("fallback",)
    assert decision.capabilities == ("fallback",)
    assert decision.confidence == 0.3
