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


def _initialize_workflow(responses=None):
    initialize_llm_client(FakeListChatModel(responses=responses or [_COMPLIANCE_PASS] * 8))
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

    assert state["skill_selection"] is None
    assert state["skill_result"] is None
    assert state["agent_assignments"] == []
    assert state["task_results"] == {}


def test_workflow_routes_retail_request_to_supervisor() -> None:
    graph = _initialize_workflow([
        '{"action":"dispatch","standalone_query":"查询商城里的手机商品",'
        '"assignments":[{"task_id":"T1","agent":"retail_agent",'
        '"objective":"查询在售手机商品","capability":"product_query",'
        '"dependencies":[],"arguments":{"query":"手机"}}],"confidence":0.99}',
        '{"action":"tool_call","tool_name":"retail_list_products",'
        '"arguments":{"query":"手机"}}',
        '{"action":"final","response":"在售商品：5G手机"}',
        '{"action":"finish","standalone_query":"查询商城里的手机商品",'
        '"response":"在售商品：5G手机","confidence":0.99}',
        _COMPLIANCE_PASS,
    ])

    result = graph.invoke(
        create_chat_state("user_001", "retail_test", "查询商城里的手机商品"),
        config={"configurable": {"thread_id": "user_001:retail_test"}},
    )

    assert result["intent"] == "retail"
    assert "在售商品：5G手机" in result["final_response"]


def test_discarded_onboarding_request_falls_back() -> None:
    graph = _initialize_workflow([
        '{"action":"finish","standalone_query":"我想开户",'
        '"response":"当前客服范围不支持开户，请说明要查询或办理的具体业务。","confidence":0.98}',
        _COMPLIANCE_PASS,
    ])

    result = graph.invoke(
        create_chat_state("user_001", "onboarding_test", "我想开户，请一步步告诉我该怎么做"),
        config={"configurable": {"thread_id": "user_001:onboarding_test"}},
    )

    assert result["intent"] == "fallback"
    assert "请说明要查询或办理的具体业务" in result["final_response"]


def test_workflow_routes_greeting_to_supervisor_fallback() -> None:
    graph = _initialize_workflow([
        '{"action":"finish","standalone_query":"你好",'
        '"response":"你好，请说明要查询或办理的具体业务。","confidence":1}',
        _COMPLIANCE_PASS,
    ])

    result = graph.invoke(
        create_chat_state("user_001", "fallback_test", "你好"),
        config={"configurable": {"thread_id": "user_001:fallback_test"}},
    )

    assert result["intent"] == "fallback"
    assert result["current_agent"] == "response_synthesizer"
