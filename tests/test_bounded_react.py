import asyncio
from types import SimpleNamespace

from domain.customer_service_agent.orchestration import bounded_react
from domain.customer_service_agent.orchestration.bounded_react import ReActLimits
from domain.customer_service_agent.orchestration.models import RouteDecision, TaskSpec
from domain.customer_service_agent.orchestration.planner import build_task_plan
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from domain.shared.identity import RequestIdentityContext


class ResponseQueue:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)

    def __call__(self, *_args, **_kwargs):
        return SimpleNamespace(content=self.contents.pop(0))


class Actions:
    def __init__(self) -> None:
        self.server = get_mcp_server()
        self.calls = []

    async def execute_read(self, tool_name, arguments, identity):
        self.calls.append((tool_name, arguments, identity.user_id))
        if tool_name == "telecom_get_current_plan":
            return {"name": "畅享套餐", "monthly_price": "99.00"}
        return []


def _identity() -> RequestIdentityContext:
    return RequestIdentityContext(user_id="user-1", session_id="session-1", turn_id="turn-1")


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="T1",
        domain="telecom",
        capability="current_plan",
        execution_mode="react",
    )


def test_bounded_react_uses_read_tool_then_returns_grounded_answer(monkeypatch) -> None:
    actions = Actions()
    monkeypatch.setattr(bounded_react, "get_action_service", lambda: actions)
    monkeypatch.setattr(bounded_react, "invoke_llm", ResponseQueue(
        '{"type":"tool","tool_name":"telecom_get_current_plan","arguments":{}}',
        '{"type":"final","answer":"当前套餐为畅享套餐，月租 99 元。"}',
    ))

    result = asyncio.run(bounded_react.execute_bounded_react(_task(), "结合费用分析当前套餐", _identity()))

    assert result.status == "succeeded"
    assert result.user_fragment == "当前套餐为畅享套餐，月租 99 元。"
    assert result.facts["tool_calls"] == 1
    assert actions.calls == [("telecom_get_current_plan", {}, "user-1")]


def test_bounded_react_blocks_write_tool_before_execution(monkeypatch) -> None:
    actions = Actions()
    monkeypatch.setattr(bounded_react, "get_action_service", lambda: actions)
    monkeypatch.setattr(bounded_react, "invoke_llm", ResponseQueue(
        '{"type":"tool","tool_name":"telecom_change_plan","arguments":{"line_id":"L1","plan_id":"P2","expected_version":1}}',
    ))

    result = asyncio.run(bounded_react.execute_bounded_react(_task(), "帮我分析套餐", _identity()))

    assert result.status == "failed"
    assert result.error_code == "agent.react_tool_not_allowed"
    assert actions.calls == []


def test_bounded_react_enforces_tool_call_limit(monkeypatch) -> None:
    actions = Actions()
    monkeypatch.setattr(bounded_react, "get_action_service", lambda: actions)
    monkeypatch.setattr(bounded_react, "invoke_llm", ResponseQueue(
        '{"type":"tool","tool_name":"telecom_get_current_plan","arguments":{}}',
        '{"type":"tool","tool_name":"telecom_get_usage_profile","arguments":{}}',
    ))

    result = asyncio.run(bounded_react.execute_bounded_react(
        _task(),
        "综合分析",
        _identity(),
        limits=ReActLimits(max_steps=3, max_tool_calls=1),
    ))

    assert result.status == "failed"
    assert result.error_code == "agent.react_tool_limit"
    assert len(actions.calls) == 1


def test_planner_selects_react_only_for_supported_complex_reads() -> None:
    read_plan = build_task_plan(
        "结合费用分析当前套餐",
        RouteDecision(domains=("telecom",), capabilities=("current_plan",), confidence=0.95),
        requires_planning=True,
    )
    write_plan = build_task_plan(
        "修改套餐",
        RouteDecision(domains=("telecom",), capabilities=("plan_change",), confidence=0.95),
        requires_planning=True,
    )

    assert read_plan.tasks[0].execution_mode == "react"
    assert write_plan.tasks[0].execution_mode == "direct"
    assert write_plan.tasks[0].effect == "write"
