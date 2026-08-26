import pytest

from domain.customer_service_agent.orchestration import query_understanding
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state


class Response:
    def __init__(self, content: str) -> None:
        self.content = content


def test_simple_query_uses_deterministic_fast_path(monkeypatch) -> None:
    monkeypatch.setattr(
        query_understanding,
        "invoke_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not run")),
    )
    state = create_chat_state("user-1", "session-1", "查询流量")

    result = query_understanding.understand_query(state)

    assert result.source == "deterministic"
    assert result.capabilities == ("usage",)
    assert result.standalone_query == "查询流量"


def test_ambiguous_order_query_uses_structured_llm_and_deterministic_time(monkeypatch) -> None:
    monkeypatch.setattr(
        query_understanding,
        "invoke_llm",
        lambda *_args, **_kwargs: Response(
            '{"standalone_query":"申请退回昨天购买的手机订单",'
            '"domains":["retail"],"capabilities":["request_return"],'
            '"entities":{"product_query":"手机","ordinal":"第二个","order_id":"invented"},'
            '"ambiguity":true,"missing_fields":["order_id"],"confidence":0.96}'
        ),
    )
    state = create_chat_state("user-1", "session-1", "把昨天买的第二个手机退掉")
    state["current_time"] = "2026-08-25"

    result = query_understanding.understand_query(state)

    assert result.source == "llm"
    assert result.capabilities == ("request_return",)
    assert result.entities == {"product_query": "手机", "ordinal": "第二个"}
    assert result.temporal_range == {"start": "2026-08-24", "end": "2026-08-25"}
    assert result.ambiguity is True


def test_explicit_resource_id_is_preserved(monkeypatch) -> None:
    monkeypatch.setattr(
        query_understanding,
        "invoke_llm",
        lambda *_args, **_kwargs: Response(
            '{"standalone_query":"查询订单","domains":["retail"],'
            '"capabilities":["order_query"],"entities":{"order_id":"order-1"},'
            '"ambiguity":false,"missing_fields":[],"confidence":0.99}'
        ),
    )
    state = create_chat_state("user-1", "session-1", "查询 order_id:order-1")

    result = query_understanding.understand_query(state)

    assert result.entities["order_id"] == "order-1"


@pytest.mark.parametrize("message", ["帮我退掉昨天购买的手机", "帮我推掉昨天购买的手机"])
def test_colloquial_return_falls_back_to_safe_deterministic_route_when_llm_is_empty(monkeypatch, message) -> None:
    monkeypatch.setattr(query_understanding, "invoke_llm", lambda *_args, **_kwargs: Response(""))
    state = create_chat_state("user-1", "session-1", message)
    state["current_time"] = "2026-08-25"

    result = query_understanding.understand_query(state)

    assert result.source == "fallback"
    assert result.domains == ("retail",)
    assert result.capabilities == ("request_return",)
    assert result.temporal_range == {"start": "2026-08-24", "end": "2026-08-25"}
