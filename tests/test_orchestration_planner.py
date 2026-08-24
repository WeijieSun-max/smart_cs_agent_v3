from domain.customer_service_agent.orchestration.planner import build_task_plan, extract_entities
from domain.customer_service_agent.orchestration.router import route_request


def test_extract_entities_reads_supported_resource_ids() -> None:
    entities = extract_entities(
        "line_id:lin-1 plan_id=plan_2 order_id：ord-3 address_id:addr-4 payment_method_id:pay-5"
    )

    assert entities == {
        "line_id": "lin-1",
        "plan_id": "plan_2",
        "order_id": "ord-3",
        "address_id": "addr-4",
        "payment_method_id": "pay-5",
    }


def test_build_task_plan_preserves_capability_effect_and_domain() -> None:
    decision = route_request("查询当前套餐并取消订单 order_id:ord-1")

    plan = build_task_plan("查询当前套餐并取消订单 order_id:ord-1", decision)
    plan.validate_dag()

    assert [task.task_id for task in plan.tasks] == ["T1", "T2"]
    assert [task.domain for task in plan.tasks] == ["telecom", "retail"]
    assert [task.capability for task in plan.tasks] == ["current_plan", "cancel_order"]
    assert [task.effect for task in plan.tasks] == ["read", "write"]
    assert plan.tasks[1].arguments["order_id"] == "ord-1"


def test_build_task_plan_uses_shared_domain_for_fallback() -> None:
    decision = route_request("你好")

    plan = build_task_plan("你好", decision)

    assert len(plan.tasks) == 1
    assert plan.tasks[0].domain == "shared"
    assert plan.tasks[0].capability == "fallback"
    assert plan.tasks[0].effect == "read"
