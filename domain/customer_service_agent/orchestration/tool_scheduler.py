from __future__ import annotations

import asyncio
import json
from typing import Any

from domain.action_governance import get_action_service
from domain.customer_service_agent.orchestration.models import AgentResult, TaskPlan, TaskSpec
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.identity import RequestIdentityContext
from pkg.telemetry import normalize_error


CAPABILITY_TO_TOOL = {
    "current_plan": "telecom_get_current_plan",
    "usage": "telecom_get_usage_profile",
    "order_query": "retail_get_order",
    "product_query": "retail_list_products",
}


async def execute_composite(
    plan: TaskPlan,
    state: ChatState,
    identity: RequestIdentityContext,
) -> list[AgentResult]:
    del state
    actions = get_action_service() if any(task.effect == "read" for task in plan.tasks) else None
    parallel: list[TaskSpec] = []
    serial: list[TaskSpec] = []
    writes: list[TaskSpec] = []
    for task in plan.tasks:
        if task.effect == "write":
            writes.append(task)
            continue
        definition = _tool_definition(task, actions)
        if definition is not None and definition.parallel_safe and not task.dependencies:
            parallel.append(task)
        else:
            serial.append(task)

    by_task_id: dict[str, AgentResult] = {}
    if parallel:
        parallel_results = await asyncio.gather(
            *(execute_read_task(task, identity, actions=actions) for task in parallel)
        )
        by_task_id.update((result.task_id, result) for result in parallel_results)
    for task in serial:
        result = await execute_read_task(task, identity, actions=actions)
        by_task_id[result.task_id] = result
    for task in writes:
        by_task_id[task.task_id] = AgentResult(
            task_id=task.task_id,
            status="skipped",
            user_fragment="复合请求中的写操作必须逐项展示影响并分别确认；请在只读结果后单独发起该项办理。",
            error_code="action.separate_confirmation_required",
        )
    return [by_task_id[task.task_id] for task in plan.tasks if task.task_id in by_task_id]


async def execute_read_task(
    task: TaskSpec,
    identity: RequestIdentityContext,
    *,
    actions=None,
) -> AgentResult:
    if task.effect != "read":
        return _failed(task, "tool.read_effect_required", "该任务不是可执行的只读任务。")
    tool_name = CAPABILITY_TO_TOOL.get(task.capability)
    if tool_name is None:
        return _failed(task, "tool.capability_not_supported", "当前只读能力暂不可用。")
    arguments_result = _arguments(task)
    if isinstance(arguments_result, AgentResult):
        return arguments_result
    actions = actions or get_action_service()
    definition = _tool_definition(task, actions)
    if definition is None:
        return _failed(task, "tool.metadata_invalid", "当前只读能力配置无效。")
    try:
        result = await actions.execute_read(tool_name, arguments_result, identity)
        return AgentResult(
            task_id=task.task_id,
            status="succeeded",
            facts={"result": result},
            user_fragment=_facts_text(task.capability, result),
        )
    except Exception as exc:
        error = normalize_error(exc)
        return _failed(
            task,
            str(error["error_code"]),
            "该项信息暂时无法查询，其他已成功结果仍然有效。",
        )


def _tool_definition(task: TaskSpec, actions):
    tool_name = CAPABILITY_TO_TOOL.get(task.capability)
    if tool_name is None:
        return None
    definition = actions.server.get_tool(tool_name)
    if definition is None or definition.effect != "read":
        return None
    if definition.capabilities and task.capability not in definition.capabilities:
        return None
    return definition


def _arguments(task: TaskSpec) -> dict[str, Any] | AgentResult:
    arguments: dict[str, Any] = {}
    if task.capability in {"current_plan", "usage"} and task.arguments.get("line_id"):
        arguments["line_id"] = task.arguments["line_id"]
    if task.capability == "order_query":
        if not task.arguments.get("order_id"):
            return _failed(task, "entity.order_id_required", "请提供 order_id。")
        arguments["order_id"] = task.arguments["order_id"]
    if task.capability == "product_query":
        arguments["query"] = ""
    return arguments


def _failed(task: TaskSpec, error_code: str, text: str) -> AgentResult:
    return AgentResult(
        task_id=task.task_id,
        status="failed",
        user_fragment=text,
        error_code=error_code,
    )


def _facts_text(capability: str, result: Any) -> str:
    if capability == "current_plan":
        return f"当前套餐：{result['name']}，月租 {result['monthly_price']} {result.get('currency', 'CNY')}，线路版本 {result['line_version']}。"
    if capability == "usage":
        return f"{result['analysis_period']}：平均流量 {result['average_data_mb']}MB、峰值 {result['peak_data_mb']}MB；平均通话 {result['average_voice_minutes']}分钟、峰值 {result['peak_voice_minutes']}分钟。数据质量：{result['data_quality']}。"
    if capability == "order_query":
        return f"订单 {result.get('order_no', result['order_id'])} 当前状态：{result['status']}，总额 {result.get('grand_total')} {result.get('currency', 'CNY')}。"
    if capability == "product_query":
        return "在售商品：" + "、".join(str(item.get("name")) for item in result[:10]) if result else "当前没有匹配的在售商品。"
    return json.dumps(result, ensure_ascii=False, default=str)
