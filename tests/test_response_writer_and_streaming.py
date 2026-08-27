from types import SimpleNamespace

import pytest

from application.customer_service.stream_events import response_delta_chunks
from domain.customer_service_agent.orchestration.models import AgentResult
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.customer_service_agent.workflow.nodes import response_writer_node
from domain.customer_service_agent.workflow.nodes.compliance_checker_node import compliance_checker_node
from domain.customer_service_agent.workflow.nodes.response_synthesizer_node import response_synthesizer_node


def _state_with_results(*fragments: str):
    state = create_chat_state("user-1", "session-1", "查询多个业务")
    state["task_results"] = {
        f"T{index}": AgentResult(
            task_id=f"T{index}",
            status="succeeded",
            user_fragment=fragment,
        ).model_dump(mode="json")
        for index, fragment in enumerate(fragments, start=1)
    }
    state["sub_results"] = {"supervisor": "\n\n".join(fragments)}
    return state


def test_single_result_preserves_grounded_text_without_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        response_writer_node,
        "invoke_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("writer should not run")),
    )
    state = _state_with_results("知识库回答。[1]")

    result = response_writer_node.response_writer_node(state)

    assert result["draft_response"] == "知识库回答。[1]"
    assert result["node_logs"] == ["Response draft completed: deterministic"]


def test_multiple_agent_results_use_response_writer(monkeypatch) -> None:
    monkeypatch.setattr(
        response_writer_node,
        "invoke_llm",
        lambda *_args, **_kwargs: SimpleNamespace(content='{"response":"套餐信息。\\n\\n订单信息。"}'),
    )
    state = _state_with_results("套餐信息。", "订单信息。")

    result = response_writer_node.response_writer_node(state)

    assert result["draft_response"] == "套餐信息。\n\n订单信息。"
    assert result["node_logs"] == ["Response draft completed: llm"]


def test_unsafe_writer_output_is_blocked_before_final_emission(monkeypatch) -> None:
    monkeypatch.setattr(
        response_writer_node,
        "invoke_llm",
        lambda *_args, **_kwargs: SimpleNamespace(content='{"response":"该方案保证收益。"}'),
    )
    state = _state_with_results("套餐信息。", "订单信息。")
    state.update(response_writer_node.response_writer_node(state))

    compliance = compliance_checker_node(state)
    state.update(compliance)
    final = response_synthesizer_node(state)

    assert compliance["compliance_passed"] is False
    assert final["final_response"] == "抱歉，您的请求或回复内容涉及敏感信息，已转交人工客服处理。"
    assert "保证收益" not in final["final_response"]


def test_response_chunks_are_incremental_and_complete() -> None:
    content = "abcdefghij"

    chunks = response_delta_chunks(content, max_chars=4)

    assert chunks == ["abcd", "efgh", "ij"]
    assert "".join(chunks) == content


def test_response_chunks_handle_empty_content_and_invalid_size() -> None:
    assert response_delta_chunks("") == []
    with pytest.raises(ValueError, match="max_chars must be positive"):
        response_delta_chunks("answer", max_chars=0)
