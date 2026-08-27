from domain.customer_service_agent.orchestration.models import RouteDecision
from domain.customer_service_agent.orchestration.planner import build_task_plan, extract_entities


def test_text_entity_extraction_is_removed() -> None:
    assert extract_entities("order_id:order-1 amount:20") == {}


def test_typed_entities_are_preserved_without_parsing_query() -> None:
    decision = RouteDecision(
        domains=("telecom", "retail"),
        capabilities=("current_plan", "cancel_order"),
        confidence=0.99,
        composite=True,
        risk_level="medium",
    )

    plan = build_task_plan(
        "任意自然语言，不再参与参数解析",
        decision,
        entities={"order_id": "order-1", "reason": "changed_mind"},
    )
    plan.validate_dag()

    assert [task.domain for task in plan.tasks] == ["telecom", "retail"]
    assert [task.effect for task in plan.tasks] == ["read", "write"]
    assert plan.tasks[1].arguments["order_id"] == "order-1"


def test_non_tool_knowledge_capability_uses_knowledge_domain() -> None:
    decision = RouteDecision(
        domains=("retail",),
        capabilities=("retail_policy",),
        confidence=0.99,
    )

    plan = build_task_plan("查询退货政策", decision)

    assert plan.tasks[0].domain == "knowledge"
