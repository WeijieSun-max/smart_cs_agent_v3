import asyncio
import json

from domain.customer_service_agent.agents import supervisor_agent
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state


class Response:
    def __init__(self, content: str) -> None:
        self.content = content


def test_simple_query_always_uses_llm_supervisor(monkeypatch) -> None:
    calls = []

    def invoke(*args, **kwargs):
        calls.append((args, kwargs))
        return Response(
            '{"action":"dispatch","standalone_query":"查询当前套餐",'
            '"assignments":[{"task_id":"T1","agent":"telecom_agent",'
            '"objective":"查询当前用户套餐","capability":"current_plan",'
            '"dependencies":[],"arguments":{}}],"confidence":0.99}'
        )

    monkeypatch.setattr(supervisor_agent, "invoke_llm", invoke)

    decision = asyncio.run(supervisor_agent.decide_next_step(
        create_chat_state("user-1", "session-1", "查询当前套餐"),
        active_action=None,
        allow_dispatch=True,
    ))

    assert len(calls) == 1
    assert decision.action == "dispatch"
    assert decision.assignments[0].agent == "telecom_agent"
    assert decision.assignments[0].capability == "current_plan"


def test_policy_question_is_assigned_to_knowledge_agent(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor_agent,
        "invoke_llm",
        lambda *_args, **_kwargs: Response(
            '{"action":"dispatch","standalone_query":"超过7天是否可以退货",'
            '"assignments":[{"task_id":"T1","agent":"knowledge_agent",'
            '"objective":"查询退货期限政策","capability":"retail_policy",'
            '"dependencies":[],"arguments":{}}],"confidence":0.98}'
        ),
    )

    decision = asyncio.run(supervisor_agent.decide_next_step(
        create_chat_state("user-1", "session-1", "购买超过7天还能退货吗"),
        active_action=None,
        allow_dispatch=True,
    ))

    assert decision.assignments[0].agent == "knowledge_agent"
    assert decision.assignments[0].capability == "retail_policy"


def test_cross_domain_assignments_are_preserved(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor_agent,
        "invoke_llm",
        lambda *_args, **_kwargs: Response(
            '{"action":"dispatch","standalone_query":"查询套餐和订单",'
            '"assignments":['
            '{"task_id":"T1","agent":"telecom_agent","objective":"查询套餐",'
            '"capability":"current_plan","dependencies":[],"arguments":{}},'
            '{"task_id":"T2","agent":"retail_agent","objective":"查询订单",'
            '"capability":"order_query","dependencies":[],"arguments":{"order_id":"order-1"}}'
            '],"confidence":0.97}'
        ),
    )

    decision = asyncio.run(supervisor_agent.decide_next_step(
        create_chat_state("user-1", "session-1", "查询套餐和订单"),
        active_action=None,
        allow_dispatch=True,
    ))

    assert [item.agent for item in decision.assignments] == ["telecom_agent", "retail_agent"]


def test_model_failure_does_not_fall_back_to_keyword_routing(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor_agent,
        "invoke_llm",
        lambda *_args, **_kwargs: Response(""),
    )

    decision = asyncio.run(supervisor_agent.decide_next_step(
        create_chat_state("user-1", "session-1", "取消订单 order-1"),
        active_action=None,
        allow_dispatch=True,
    ))

    assert decision.action == "finish"
    assert decision.assignments == ()
    assert "模型暂时无法" in (decision.response or "")


def test_pending_action_is_classified_by_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor_agent,
        "invoke_llm",
        lambda *_args, **_kwargs: Response(
            '{"action":"confirm_action","standalone_query":"确认执行",'
            '"assignments":[],"confidence":1}'
        ),
    )

    decision = asyncio.run(supervisor_agent.decide_next_step(
        create_chat_state("user-1", "session-1", "可以，按刚才的办"),
        active_action={"action_id": "a1", "impact_summary": "变更套餐"},
        allow_dispatch=True,
    ))

    assert decision.action == "confirm_action"


def test_pending_task_is_given_to_supervisor_for_standalone_assignment(monkeypatch) -> None:
    captured = []

    def invoke(messages, **_kwargs):
        captured.extend(messages)
        return Response(
            '{"action":"dispatch","standalone_query":"创建北京默认地址并沿用原联系人",'
            '"assignments":[{"task_id":"R2","agent":"retail_agent",'
            '"objective":"创建北京市朝阳区应天路88号并沿用默认联系人",'
            '"capability":"default_address","dependencies":[],'
            '"arguments":{"contact_strategy":"reuse_current_default"}}],"confidence":0.99}'
        )

    monkeypatch.setattr(supervisor_agent, "invoke_llm", invoke)
    state = create_chat_state(
        "user-1",
        "session-1",
        "使用原来默认地址的联系人",
        pending_task={
            "task_id": "R1",
            "agent": "retail_agent",
            "capability": "default_address",
            "objective": "创建北京市朝阳区应天路88号并设为默认地址",
            "arguments": {"detail": "应天路88号"},
        },
    )

    decision = asyncio.run(supervisor_agent.decide_next_step(
        state,
        active_action=None,
        allow_dispatch=True,
    ))

    payload = json.loads(captured[-1].content)
    assert payload["active_pending_task"]["arguments"]["detail"] == "应天路88号"
    assert decision.assignments[0].arguments["contact_strategy"] == "reuse_current_default"
