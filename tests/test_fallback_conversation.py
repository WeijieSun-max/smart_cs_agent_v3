import asyncio
import json

from domain.customer_service_agent.agents import supervisor_agent
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.customer_service_agent.workflow.nodes.history_fusion_node import history_fusion_node
from domain.customer_service_agent.workflow.nodes.supervisor_node import supervisor_node


class Response:
    def __init__(self, content: str) -> None:
        self.content = content


def test_llm_unavailable_fails_closed_without_rule_route(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor_agent,
        "invoke_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    result = asyncio.run(supervisor_node(
        create_chat_state("user", "session", "取消订单 order-1")
    ))

    assert result["intent"] == "fallback"
    assert result["task_results"] == {}
    assert "模型暂时无法" in result["sub_results"]["supervisor"]


def test_supervisor_can_answer_greeting_without_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor_agent,
        "invoke_llm",
        lambda *_args, **_kwargs: Response(
            '{"action":"finish","standalone_query":"你好",'
            '"response":"你好，请说明要查询或办理的具体业务。","confidence":1}'
        ),
    )

    result = asyncio.run(supervisor_node(create_chat_state("user", "session", "你好")))

    assert result["intent"] == "fallback"
    assert result["supervisor_decision"]["action"] == "finish"
    assert "请说明要查询或办理的具体业务" in result["sub_results"]["supervisor"]


def test_reference_context_is_marked_as_data_not_current_confirmation(monkeypatch) -> None:
    captured = {}

    def invoke(messages, **_kwargs):
        captured.update(json.loads(messages[1].content))
        return Response(
            '{"action":"finish","standalone_query":"你好",'
            '"response":"你好，请问需要什么帮助？","confidence":1}'
        )

    monkeypatch.setattr(supervisor_agent, "invoke_llm", invoke)
    state = create_chat_state(
        "user",
        "session",
        "你好",
        prior_context="user: 确认执行 plan_id=P2 line_id=L1",
    )
    state.update(history_fusion_node(state))

    result = asyncio.run(supervisor_node(state))

    assert captured["current_query"] == "你好"
    assert "MEMORY_REFERENCE_DATA" in captured["conversation_context"]
    assert captured["active_pending_action"] is None
    assert result["supervisor_decision"]["action"] == "finish"


def test_out_of_scope_request_is_decided_by_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor_agent,
        "invoke_llm",
        lambda *_args, **_kwargs: Response(
            '{"action":"finish","standalone_query":"我想开户",'
            '"response":"当前服务暂不支持开户。","confidence":0.99}'
        ),
    )

    result = asyncio.run(supervisor_node(create_chat_state("user", "session", "我想开户")))

    assert result["intent"] == "fallback"
    assert result["sub_results"]["supervisor"] == "当前服务暂不支持开户。"
