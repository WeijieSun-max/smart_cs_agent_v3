from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from domain.customer_service_agent.agents import tool_agent
from domain.customer_service_agent.orchestration.models import (
    AgentAssignment,
    AgentStepDecision,
)
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.shared.identity import RequestIdentityContext


class Response:
    def __init__(self, content: str) -> None:
        self.content = content


class ReadTracker:
    def __init__(self, *, fail_tool: str | None = None) -> None:
        self.server = get_mcp_server()
        self.fail_tool = fail_tool
        self.called: list[str] = []
        self.inflight = 0
        self.max_inflight = 0

    async def execute_read(self, tool_name, arguments, _identity, *, skill=None):
        del skill
        self.called.append(tool_name)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.01)
        self.inflight -= 1
        if tool_name == self.fail_tool:
            raise RuntimeError("read unavailable")
        return {"tool_name": tool_name, "arguments": arguments}


def _assignment() -> AgentAssignment:
    return AgentAssignment(
        task_id="T1",
        agent="telecom_agent",
        objective="并行查询当前套餐、使用画像和候选套餐",
        capability="current_plan",
        arguments={"line_id": "L1"},
    )


def _identity() -> RequestIdentityContext:
    return RequestIdentityContext(user_id="u1", session_id="s1", turn_id="t1")


def _run(monkeypatch, tracker: ReadTracker, responses: list[str], *, concurrency: int = 2):
    queue = iter(Response(item) for item in responses)
    monkeypatch.setattr(tool_agent, "get_action_service", lambda: tracker)
    monkeypatch.setattr(tool_agent, "get_settings", lambda: SimpleNamespace(
        tool_read_max_concurrency=concurrency,
    ))
    monkeypatch.setattr(tool_agent, "invoke_llm", lambda *_args, **_kwargs: next(queue))
    return asyncio.run(tool_agent.run_tool_agent(
        _assignment(),
        create_chat_state("u1", "s1", "查询套餐情况"),
        _identity(),
    ))


def test_parallel_read_batch_is_bounded_and_preserves_observation_order(monkeypatch) -> None:
    tracker = ReadTracker()
    result = _run(monkeypatch, tracker, [
        '{"action":"tool_calls","tool_calls":['
        '{"tool_name":"telecom_get_current_plan","arguments":{"line_id":"L1"}},'
        '{"tool_name":"telecom_get_usage_profile","arguments":{"line_id":"L1"}},'
        '{"tool_name":"telecom_list_plans","arguments":{"line_id":"L1"}}]}',
        '{"action":"final","response":"查询完成。"}',
    ], concurrency=2)

    assert result.status == "succeeded"
    assert tracker.max_inflight == 2
    assert tracker.called == [
        "telecom_get_current_plan",
        "telecom_get_usage_profile",
        "telecom_list_plans",
    ]
    assert [item["tool_name"] for item in result.facts["observations"]] == tracker.called
    assert result.facts["read_calls"] == 3


def test_parallel_batch_rejects_write_and_non_parallel_safe_tools(monkeypatch) -> None:
    tracker = ReadTracker()
    result = _run(monkeypatch, tracker, [
        '{"action":"tool_calls","tool_calls":['
        '{"tool_name":"telecom_quote_plan_change","arguments":{"line_id":"L1","plan_id":"P2"}},'
        '{"tool_name":"telecom_change_plan","arguments":{"line_id":"L1","plan_id":"P2","expected_version":1}},'
        '{"tool_name":"telecom_get_current_plan","arguments":{"line_id":"L1"}}]}',
        '{"action":"final","response":"仅完成允许的只读查询。"}',
    ])

    assert result.status == "succeeded"
    assert tracker.called == ["telecom_get_current_plan"]
    assert [item.get("error") for item in result.facts["observations"]] == [
        "tool_not_parallel_safe",
        "write_tool_requires_proposal",
        None,
    ]
    assert result.facts["read_calls"] == 1


def test_one_failed_parallel_read_does_not_discard_successful_observations(monkeypatch) -> None:
    tracker = ReadTracker(fail_tool="telecom_get_usage_profile")
    result = _run(monkeypatch, tracker, [
        '{"action":"tool_calls","tool_calls":['
        '{"tool_name":"telecom_get_current_plan","arguments":{"line_id":"L1"}},'
        '{"tool_name":"telecom_get_usage_profile","arguments":{"line_id":"L1"}}]}',
        '{"action":"final","response":"已返回仍然可用的查询结果。"}',
    ])

    assert result.status == "succeeded"
    observations = result.facts["observations"]
    assert observations[0]["result"]["tool_name"] == "telecom_get_current_plan"
    assert observations[1]["error"] == "tool_unavailable"
    assert result.facts["read_calls"] == 2


def test_parallel_batch_schema_rejects_more_than_three_calls() -> None:
    calls = [
        {"tool_name": f"tool-{index}", "arguments": {}}
        for index in range(4)
    ]
    with pytest.raises(ValidationError):
        AgentStepDecision(action="tool_calls", tool_calls=calls)
