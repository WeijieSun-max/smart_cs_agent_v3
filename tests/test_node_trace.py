from __future__ import annotations

import asyncio
import json

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from application.customer_service import agent_run_service, chat_service
from application.customer_service.agent_run_service import AgentRun
from application.customer_service.node_trace import NodeTraceRecorder, serialize_debug_value
from pkg.telemetry import create_turn_trace


def _node_event(event_type: str, run_id: str, node_name: str, value) -> dict:
    data_key = "input" if event_type.endswith("start") else "output"
    return {
        "event": event_type,
        "name": node_name,
        "run_id": run_id,
        "metadata": {"langgraph_node": node_name},
        "parent_ids": [],
        "data": {data_key: value},
    }


def test_debug_serializer_preserves_messages_and_redacts_only_credentials() -> None:
    value = {
        "messages": [
            SystemMessage(content="system prompt"),
            HumanMessage(content="phone 13800138000 email user@example.com"),
            AIMessage(content="answer"),
        ],
        "authorization": "Bearer top-secret-token",
        "OPENAI_API_KEY": "provider-secret",
        "nested": {"api_key": "sk-secretvalue123", "token_usage": {"prompt_tokens": 12}},
        "text": "password=hunter2 and Bearer abcdefghijklmnop and Basic YWxhZGRpbjpvcGVuc2VzYW1l",
    }

    result = serialize_debug_value(value)

    assert result["messages"][0]["content"] == "system prompt"
    assert "13800138000" in result["messages"][1]["content"]
    assert "user@example.com" in result["messages"][1]["content"]
    assert result["authorization"] == "[REDACTED_CREDENTIAL]"
    assert result["OPENAI_API_KEY"] == "[REDACTED_CREDENTIAL]"
    assert result["nested"]["api_key"] == "[REDACTED_CREDENTIAL]"
    assert result["nested"]["token_usage"]["prompt_tokens"] == 12
    assert "hunter2" not in result["text"]
    assert "abcdefghijklmnop" not in result["text"]
    assert "YWxhZGRpbjpvcGVuc2VzYW1l" not in result["text"]


def test_debug_serializer_handles_cycles_and_size_limit() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    cycle_result = serialize_debug_value(cyclic)
    size_result = serialize_debug_value("x" * 100, max_bytes=20)

    assert cycle_result["self"]["_debug_error"] == "CIRCULAR_REFERENCE"
    assert size_result["_debug_truncated"] is True
    assert size_result["original_bytes"] > 20


def test_debug_serializer_records_only_memory_packet_metadata() -> None:
    secret_memory = "用户只接受短信联系"
    value = {
        "memory_packet": {
            "session_summary": "用户曾咨询退款",
            "recent_messages": [{"role": "user", "content": "退款什么时候到"}],
            "episodes": [],
            "semantic_memories": [
                {"memory_type": "preference", "content": secret_memory},
            ],
            "token_count": 42,
            "max_tokens": 1800,
        },
        "context_text": (
            "<<<MEMORY_REFERENCE_DATA>>>\n"
            f"{secret_memory}\n"
            "<<<END_MEMORY_REFERENCE_DATA>>>"
        ),
    }

    result = serialize_debug_value(value)

    assert result["memory_packet"] == {
        "session_summary_present": True,
        "recent_message_count": 1,
        "episode_count": 0,
        "semantic_memory_count": 1,
        "memory_types": ["preference"],
        "token_count": 42,
        "max_tokens": 1800,
    }
    assert secret_memory not in json.dumps(result, ensure_ascii=False)
    assert result["context_text"] == "[REDACTED_MEMORY_REFERENCE_DATA]"


def test_node_trace_recorder_maps_node_and_multiple_model_calls() -> None:
    recorder = NodeTraceRecorder("turn-1")

    node_start = recorder.consume(_node_event("on_chain_start", "node-1", "supervisor_node", {"raw_query": "hello"}))
    model_1_start = recorder.consume({
        "event": "on_chat_model_start",
        "name": "ChatOpenAI",
        "run_id": "model-1",
        "metadata": {"langgraph_node": "supervisor_node"},
        "parent_ids": ["root", "node-1"],
        "data": {"input": {"messages": [[SystemMessage(content="route"), HumanMessage(content="hello")]]}},
    })
    model_1_end = recorder.consume({
        "event": "on_chat_model_end",
        "name": "ChatOpenAI",
        "run_id": "model-1",
        "metadata": {"langgraph_node": "supervisor_node"},
        "parent_ids": ["root", "node-1"],
        "data": {"output": AIMessage(content='{"intent":"knowledge"}')},
    })
    model_2_start = recorder.consume({
        "event": "on_chat_model_start",
        "name": "ChatOpenAI",
        "run_id": "model-2",
        "metadata": {"langgraph_node": "supervisor_node"},
        "parent_ids": ["node-1"],
        "data": {"input": {"messages": [[HumanMessage(content="second")]]}},
    })
    model_2_end = recorder.consume({
        "event": "on_chat_model_end",
        "name": "ChatOpenAI",
        "run_id": "model-2",
        "metadata": {"langgraph_node": "supervisor_node"},
        "parent_ids": ["node-1"],
        "data": {"output": AIMessage(content="second response")},
    })
    node_end = recorder.consume(_node_event("on_chain_end", "node-1", "supervisor_node", {"intent": "knowledge"}))

    events = [node_start, model_1_start, model_1_end, model_2_start, model_2_end, node_end]
    assert [event["phase"] for event in events] == [
        "node_start",
        "llm_start",
        "llm_end",
        "llm_start",
        "llm_end",
        "node_end",
    ]
    assert [event["sequence"] for event in events] == [1, 2, 3, 4, 5, 6]
    assert model_1_start["node_trace_id"] == "node-1"
    assert model_1_end["data"]["response"]["content"] == '{"intent":"knowledge"}'
    assert model_2_start["model_call_id"] == "model-2"
    assert node_end["data"]["output"]["intent"] == "knowledge"


def test_node_trace_recorder_uses_metadata_fallback_and_redacts_errors() -> None:
    recorder = NodeTraceRecorder("turn-2")
    recorder.consume(_node_event("on_chain_start", "node-2", "compliance_checker_node", {"query": "refund"}))

    model_start = recorder.consume({
        "event": "on_chat_model_start",
        "name": "ChatOpenAI",
        "run_id": "model-fallback",
        "metadata": {"langgraph_node": "compliance_checker_node"},
        "parent_ids": [],
        "data": {"input": "authorization=Bearer abcdefghijklmnop"},
    })
    failed = recorder.fail_active(RuntimeError("api_key=sk-secretvalue123 request failed"))

    assert model_start["node_trace_id"] == "node-2"
    assert "abcdefghijklmnop" not in model_start["data"]["prompt"]
    assert len(failed) == 1
    assert failed[0]["phase"] == "node_error"
    assert "secretvalue123" not in failed[0]["data"]["error"]["message"]


def test_dynamic_skill_model_trace_is_fully_redacted() -> None:
    recorder = NodeTraceRecorder("turn-private")
    recorder.consume(
        _node_event(
            "on_chain_start",
            "node-private",
            "dynamic_skill_agent_node",
            {"raw_query": "open an account"},
        )
    )
    started = recorder.consume(
        {
            "event": "on_chat_model_start",
            "name": "ChatOpenAI",
            "run_id": "model-private",
            "metadata": {"langgraph_node": "dynamic_skill_agent_node"},
            "parent_ids": ["node-private"],
            "data": {
                "input": {
                    "messages": [[HumanMessage(content="raw-profile-value")]]
                }
            },
        }
    )
    ended = recorder.consume(
        {
            "event": "on_chat_model_end",
            "name": "ChatOpenAI",
            "run_id": "model-private",
            "metadata": {"langgraph_node": "dynamic_skill_agent_node"},
            "parent_ids": ["node-private"],
            "data": {"output": AIMessage(content="raw-observation-value")},
        }
    )

    assert "raw-profile-value" not in repr(started)
    assert "raw-observation-value" not in repr(ended)
    assert started["data"]["prompt"] == {
        "redacted": True,
        "reason": "dynamic_skill_payload",
    }
    assert ended["data"]["response"] == {
        "redacted": True,
        "reason": "dynamic_skill_payload",
    }


def test_dynamic_skill_node_end_is_an_entire_fixed_summary() -> None:
    recorder = NodeTraceRecorder("turn-safe-output")
    recorder.consume(
        _node_event(
            "on_chain_start", "node-safe-output", "dynamic_skill_agent_node", {}
        )
    )
    ended = recorder.consume(
        _node_event(
            "on_chain_end",
            "node-safe-output",
            "dynamic_skill_agent_node",
            {
                "skill_selection": {
                    "selected_skill": "onboarding_material_check",
                    "confidence": 0.9,
                    "reason": "must-not-trace",
                },
                "skill_result": {
                    "status": "completed",
                    "skill_name": "onboarding_material_check",
                    "response": "must-not-trace",
                    "tool_calls": [
                        {
                            "tool_name": "user_profile",
                            "success": True,
                            "arguments": {"secret": "must-not-trace"},
                        }
                    ],
                    "budget_usage": {"runtime_llm_decisions": 1, "tool_calls": 1},
                },
                "sub_results": {"dynamic_skill_agent": "must-not-trace"},
                "unknown": [{"response": "must-not-trace"}],
            },
        )
    )
    assert "must-not-trace" not in repr(ended)
    assert set(ended["data"]["output"]) == {"skill_selection", "skill_result"}
    assert ended["data"]["output"]["skill_result"]["tool_calls"] == [
        {"tool_name": "user_profile", "success": True}
    ]


@pytest.mark.parametrize("source", [None, [], {"event": []}, {"event": "on_custom_event", "name": []}, {"event": "on_custom_event", "name": "skill_completed", "data": []}, {"event": "on_custom_event", "name": "skill_completed", "data": {"duration_ms": 10 ** 10_000}}])
def test_skill_sse_sanitizer_never_raises_for_wrong_exact_types(source) -> None:
    result = chat_service._safe_skill_sse_event(source)
    if type(source) is dict and type(source.get("data")) is dict:
        assert result == {"type": "skill_event", "event": "skill_completed"}
    else:
        assert result is None


def test_stream_emits_live_node_trace_without_adding_raw_data_to_agent_steps(monkeypatch) -> None:
    class Graph:
        async def astream_events(self, *_args, **_kwargs):
            yield _node_event(
                "on_chain_start",
                "node-live",
                "response_generation_node",
                {"raw_query": "phone 13800138000", "api_key": "sk-secretvalue123"},
            )
            yield {
                "event": "on_chat_model_start",
                "name": "ChatOpenAI",
                "run_id": "model-live",
                "metadata": {"langgraph_node": "response_generation_node"},
                "parent_ids": ["node-live"],
                "data": {"input": {"messages": [[SystemMessage(content="full system prompt")]]}},
            }
            yield {
                "event": "on_chat_model_end",
                "name": "ChatOpenAI",
                "run_id": "model-live",
                "metadata": {"langgraph_node": "response_generation_node"},
                "parent_ids": ["node-live"],
                "data": {"output": AIMessage(content="raw model response")},
            }
            yield _node_event(
                "on_chain_end",
                "node-live",
                "response_generation_node",
                {"final_response": "final answer", "compliance_passed": True},
            )

    class Memory:
        def add_message(self, *_args) -> None:
            return None

    async def collect() -> tuple[list[dict], AgentRun]:
        turn = create_turn_trace("user-1", "session-live")
        run = AgentRun(session_id="session-live", turn_id=turn.turn_id)
        payloads = []
        async for chunk in chat_service._generate_stream_events_admitted(
            Graph(),
            {"raw_query": "phone 13800138000"},
            "session-live",
            "user-1",
            turn,
            run,
        ):
            if chunk.startswith("data:{"):
                payloads.append(json.loads(chunk.removeprefix("data:").strip()))
        return payloads, run

    monkeypatch.setattr(chat_service, "_build_graph_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(chat_service.short_term_memory_service, "get_service", lambda: Memory())
    monkeypatch.setattr(agent_run_service.registry, "_persist", lambda *_args, **_kwargs: None)

    payloads, run = asyncio.run(collect())
    traces = [payload for payload in payloads if payload["type"] == "node_trace"]

    assert [trace["phase"] for trace in traces] == ["node_start", "llm_start", "llm_end", "node_end"]
    assert traces[0]["data"]["input"]["api_key"] == "[REDACTED_CREDENTIAL]"
    assert "13800138000" in traces[0]["data"]["input"]["raw_query"]
    assert traces[1]["data"]["prompt"]["messages"][0][0]["content"] == "full system prompt"
    assert traces[2]["data"]["response"]["content"] == "raw model response"
    assert all("input" not in step and "output" not in step for step in run.steps)


def test_stream_forwards_only_rebuilt_safe_skill_event_fields(monkeypatch) -> None:
    class Graph:
        async def astream_events(self, *_args, **_kwargs):
            yield {
                "event": "on_custom_event",
                "name": "skill_tool_end",
                "data": {
                    "skill_name": "onboarding_material_check",
                    "tool_name": "user_profile",
                    "status": "success",
                    "duration_ms": 12,
                    "arguments": {"user_id": "must-not-stream"},
                    "query": "must-not-stream",
                    "source": "must-not-stream",
                },
            }
            yield _node_event(
                "on_chain_start", "node-safe", "response_synthesizer_node", {}
            )
            yield _node_event(
                "on_chain_end",
                "node-safe",
                "response_synthesizer_node",
                {"final_response": "done"},
            )

    class Memory:
        def add_message(self, *_args) -> None:
            return None

    async def collect() -> list[dict]:
        turn = create_turn_trace("user-safe", "session-safe")
        run = AgentRun(session_id="session-safe", turn_id=turn.turn_id)
        payloads = []
        async for chunk in chat_service._generate_stream_events_admitted(
            Graph(), {"raw_query": "private"}, "session-safe", "user-safe", turn, run
        ):
            if chunk.startswith("data:{"):
                payloads.append(json.loads(chunk.removeprefix("data:").strip()))
        return payloads

    monkeypatch.setattr(chat_service, "_build_graph_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(chat_service.short_term_memory_service, "get_service", lambda: Memory())
    monkeypatch.setattr(agent_run_service.registry, "_persist", lambda *_args, **_kwargs: None)

    skill_events = [
        event for event in asyncio.run(collect()) if event["type"] == "skill_event"
    ]
    assert skill_events == [
        {
            "type": "skill_event",
            "event": "skill_tool_end",
            "skill_name": "onboarding_material_check",
            "tool_name": "user_profile",
            "status": "success",
            "duration_ms": 12.0,
        }
    ]
    assert "must-not-stream" not in repr(skill_events)
