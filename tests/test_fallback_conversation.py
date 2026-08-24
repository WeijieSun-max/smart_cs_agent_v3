import asyncio

from domain.customer_service_agent.orchestration.router import route_request
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.customer_service_agent.workflow.nodes.history_fusion_node import history_fusion_node
from domain.customer_service_agent.workflow.nodes.supervisor_node import supervisor_node


def test_unknown_request_routes_to_deterministic_fallback() -> None:
    decision = route_request("随便聊聊")

    assert decision.domains == ("fallback",)
    assert decision.capabilities == ("fallback",)
    assert decision.confidence == 0.3


def test_known_telecom_request_keeps_business_route() -> None:
    decision = route_request("查询当前套餐")

    assert decision.domains == ("telecom",)
    assert decision.capabilities == ("current_plan",)


def test_cross_domain_read_is_marked_composite() -> None:
    decision = route_request("查询当前套餐和商城商品")

    assert decision.domains == ("telecom", "retail")
    assert decision.composite is True


def test_known_retail_request_keeps_business_route() -> None:
    decision = route_request("查询商城里的手机商品")

    assert decision.domains == ("retail",)
    assert decision.capabilities == ("product_query",)


def test_supervisor_returns_current_production_fallback_response() -> None:
    result = asyncio.run(supervisor_node(create_chat_state("user", "session", "你好")))

    assert result["intent"] == "fallback"
    assert result["current_agent"] == "supervisor"
    assert "请说明要查询或办理的具体业务" in result["sub_results"]["supervisor"]


def test_reference_context_cannot_trigger_confirmation_or_write_route() -> None:
    state = create_chat_state("user", "session", "你好", prior_context="user: 确认执行 plan_id=P2 line_id=L1")
    state.update(history_fusion_node(state))

    result = asyncio.run(supervisor_node(state))

    assert result["intent"] == "fallback"
    assert result["route_decision"]["capabilities"] == ["fallback"]


def test_discarded_onboarding_route_stays_in_fallback() -> None:
    result = asyncio.run(supervisor_node(create_chat_state("user", "session", "我想开户")))

    assert result["intent"] == "fallback"
    assert result["route_decision"]["domains"] == ["fallback"]
