import asyncio
from dataclasses import replace

from domain.customer_service_agent.orchestration import tool_scheduler
from domain.customer_service_agent.orchestration.models import TaskPlan, TaskSpec
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.shared.identity import RequestIdentityContext


def _identity() -> RequestIdentityContext:
    return RequestIdentityContext(
        user_id="user-1",
        session_id="session-1",
        turn_id="turn-1",
    )


class DefinitionServer:
    def __init__(self, overrides=None) -> None:
        self.overrides = overrides or {}

    def get_tool(self, name):
        definition = get_mcp_server().get_tool(name)
        changes = self.overrides.get(name)
        return replace(definition, **changes) if definition is not None and changes else definition


class ConcurrentActions:
    def __init__(self, *, overrides=None, failing_tool=None) -> None:
        self.server = DefinitionServer(overrides)
        self.started = 0
        self.inflight = 0
        self.max_inflight = 0
        self.ready = asyncio.Event()
        self.failing_tool = failing_tool

    async def execute_read(self, tool_name, arguments, identity):
        del arguments, identity
        self.started += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        if self.started == 2:
            self.ready.set()
        if self.server.get_tool(tool_name).parallel_safe:
            try:
                await asyncio.wait_for(self.ready.wait(), timeout=0.2)
            except TimeoutError:
                pass
        await asyncio.sleep(0)
        self.inflight -= 1
        if tool_name == self.failing_tool:
            raise RuntimeError("read unavailable")
        if tool_name == "telecom_get_current_plan":
            return {
                "name": "畅享套餐",
                "monthly_price": 59,
                "currency": "CNY",
                "line_version": 2,
            }
        return [{"name": "5G手机"}]


class UnexpectedActions:
    server = DefinitionServer()

    async def execute_read(self, tool_name, arguments, identity):
        raise AssertionError((tool_name, arguments, identity))


def _composite_plan(*tasks: TaskSpec) -> TaskPlan:
    return TaskPlan(tasks=tasks)


def _execute(plan: TaskPlan):
    return tool_scheduler.execute_composite(
        plan,
        create_chat_state("user-1", "session-1", "查询套餐和商品"),
        _identity(),
    )


def test_execute_composite_runs_parallel_safe_reads_concurrently(monkeypatch) -> None:
    actions = ConcurrentActions()
    monkeypatch.setattr(tool_scheduler, "get_action_service", lambda: actions)
    plan = _composite_plan(
        TaskSpec(task_id="T1", domain="telecom", capability="current_plan"),
        TaskSpec(task_id="T2", domain="retail", capability="product_query"),
    )

    results = asyncio.run(asyncio.wait_for(_execute(plan), timeout=1))

    assert actions.max_inflight == 2
    assert [result.status for result in results] == ["succeeded", "succeeded"]
    assert results[0].user_fragment.startswith("当前套餐")
    assert results[1].user_fragment == "在售商品：5G手机"


def test_non_parallel_safe_read_is_serialized(monkeypatch) -> None:
    actions = ConcurrentActions(overrides={"retail_list_products": {"parallel_safe": False}})
    monkeypatch.setattr(tool_scheduler, "get_action_service", lambda: actions)
    plan = _composite_plan(
        TaskSpec(task_id="T1", domain="telecom", capability="current_plan"),
        TaskSpec(task_id="T2", domain="retail", capability="product_query"),
    )

    results = asyncio.run(_execute(plan))

    assert actions.max_inflight == 1
    assert [result.task_id for result in results] == ["T1", "T2"]


def test_partial_read_failure_preserves_other_results(monkeypatch) -> None:
    actions = ConcurrentActions(failing_tool="retail_list_products")
    monkeypatch.setattr(tool_scheduler, "get_action_service", lambda: actions)
    plan = _composite_plan(
        TaskSpec(task_id="T1", domain="telecom", capability="current_plan"),
        TaskSpec(task_id="T2", domain="retail", capability="product_query"),
    )

    results = asyncio.run(_execute(plan))

    assert [result.status for result in results] == ["succeeded", "failed"]
    assert results[1].error_code
    assert "其他已成功结果仍然有效" in results[1].user_fragment


def test_execute_composite_skips_every_write_in_plan_order(monkeypatch) -> None:
    monkeypatch.setattr(
        tool_scheduler,
        "get_action_service",
        lambda: (_ for _ in ()).throw(AssertionError("write-only plan must not load action service")),
    )
    plan = _composite_plan(
        TaskSpec(task_id="T1", domain="retail", capability="cancel_order", effect="write"),
        TaskSpec(task_id="T2", domain="retail", capability="update_order_address", effect="write"),
    )

    results = asyncio.run(_execute(plan))

    assert [result.task_id for result in results] == ["T1", "T2"]
    assert all(result.status == "skipped" for result in results)
    assert all(result.error_code == "action.separate_confirmation_required" for result in results)


def test_order_read_requires_order_id_without_calling_tool(monkeypatch) -> None:
    monkeypatch.setattr(tool_scheduler, "get_action_service", lambda: UnexpectedActions())
    task = TaskSpec(task_id="T1", domain="retail", capability="order_query")

    result = asyncio.run(tool_scheduler.execute_read_task(task, _identity()))

    assert result.status == "failed"
    assert result.error_code == "entity.order_id_required"
    assert result.user_fragment == "请提供 order_id。"


def test_invalid_tool_metadata_fails_without_execution(monkeypatch) -> None:
    actions = ConcurrentActions(overrides={"telecom_get_current_plan": {"effect": "write"}})
    monkeypatch.setattr(tool_scheduler, "get_action_service", lambda: actions)
    task = TaskSpec(task_id="T1", domain="telecom", capability="current_plan")

    result = asyncio.run(tool_scheduler.execute_read_task(task, _identity()))

    assert result.status == "failed"
    assert result.error_code == "tool.metadata_invalid"
    assert actions.started == 0
