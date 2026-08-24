from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import MemorySaver

from domain.action_governance import GovernedActionService, initialize_action_service
from domain.business.service import initialize_service
from domain.business.store import InMemoryBusinessStore
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from domain.customer_service_agent.workflow import customer_service_workflow
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.shared.checkpoint.checkpoint_saver_service import initialize_service as initialize_checkpoint
from domain.shared.llm.llm_service import initialize_llm_client


_COMPLIANCE_PASS = '{"passed": true, "risk_level": "low", "violations": [], "suggestions": []}'


def _initialize_workflow():
    initialize_llm_client(FakeListChatModel(responses=[_COMPLIANCE_PASS] * 8))
    initialize_checkpoint(MemorySaver())
    business = initialize_service(InMemoryBusinessStore({
        "users": [{"user_id": "user_001", "status": "active"}],
        "products": [{"product_id": "P1", "name": "5G手机", "description": "测试商品", "status": "active"}],
    }))
    initialize_action_service(GovernedActionService(get_mcp_server(), business))
    customer_service_workflow.initialize_workflow()
    return customer_service_workflow.get_workflow()


def test_workflow_constructs_without_external_services() -> None:
    assert _initialize_workflow() is not None


def test_chat_state_has_current_runtime_defaults() -> None:
    state = create_chat_state("user_001", "session_001", "查询套餐")

    assert state["pending_action_intent"] == "none"
    assert state["pending_action_route"] == "route"
    assert state["skill_selection"] is None
    assert state["skill_result"] is None
    assert state["route_decision"] is None
    assert state["task_plan"] is None


def test_workflow_routes_retail_request_to_supervisor() -> None:
    graph = _initialize_workflow()

    result = graph.invoke(
        create_chat_state("user_001", "retail_test", "查询商城里的手机商品"),
        config={"configurable": {"thread_id": "user_001:retail_test"}},
    )

    assert result["intent"] == "retail"
    assert "在售商品：5G手机" in result["final_response"]


def test_discarded_onboarding_request_falls_back() -> None:
    graph = _initialize_workflow()

    result = graph.invoke(
        create_chat_state("user_001", "onboarding_test", "我想开户，请一步步告诉我该怎么做"),
        config={"configurable": {"thread_id": "user_001:onboarding_test"}},
    )

    assert result["intent"] == "fallback"
    assert "请说明要查询或办理的具体业务" in result["final_response"]


def test_workflow_routes_greeting_to_supervisor_fallback() -> None:
    graph = _initialize_workflow()

    result = graph.invoke(
        create_chat_state("user_001", "fallback_test", "你好"),
        config={"configurable": {"thread_id": "user_001:fallback_test"}},
    )

    assert result["intent"] == "fallback"
    assert result["current_agent"] == "response_synthesizer"
