"""Supervisor 主循环、依赖感知的领域分派与治理动作执行节点。"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from pydantic import ValidationError

from domain.action_governance import get_action_service
from domain.customer_service_agent.agents import (
    MAX_SUPERVISOR_ROUNDS,
    decide_next_step,
)
from domain.customer_service_agent.orchestration.capability_index import get_capability_index
from domain.customer_service_agent.orchestration.domain_agents import (
    knowledge_agent_graph,
    retail_agent_graph,
    telecom_agent_graph,
)
from domain.customer_service_agent.orchestration.models import (
    AgentAssignment,
    AgentResult,
    SupervisorDecision,
)
from domain.customer_service_agent.orchestration.pending_action_resolver import (
    explicit_pending_action_decision,
    resolve_pending_action,
)
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.identity import RequestIdentityContext
from pkg.log.logger import get_logger
from pkg.telemetry import normalize_error

logger = get_logger()


async def supervisor_manager_node(state: ChatState) -> dict[str, Any]:
    """读取活跃治理动作，让 Supervisor 决策并生成下一分支状态。

    只有 dispatch 增加轮次；finish/clarify 直接产生待写作响应。活跃治理动作
    会作为只读摘要提供给 Supervisor，并由决策校验阻止此时调度新业务。
    """

    identity = identity_from_state(state)
    active_action = await _active_action(identity)
    round_number = int(state.get("supervisor_round") or 0)
    query = (state.get("raw_query") or "").strip()
    explicit_action = (
        explicit_pending_action_decision(query)
        if active_action is not None
        else None
    )
    if explicit_action is not None:
        decision = SupervisorDecision(
            action=explicit_action,
            standalone_query=query,
            confidence=1.0,
        )
        decision_source = "deterministic_action_command"
    else:
        decision = await decide_next_step(
            state,
            active_action=active_action,
            allow_dispatch=round_number < MAX_SUPERVISOR_ROUNDS,
        )
        decision_source = "llm"
    update: dict[str, Any] = {
        "normalized_query": decision.standalone_query,
        "supervisor_decision": decision.model_dump(mode="json"),
        "active_action": active_action,
        "agent_assignments": [item.model_dump(mode="json") for item in decision.assignments],
        "current_agent": "supervisor",
        "node_logs": [
            f"Supervisor decision: {decision.action} ({decision_source})"
        ],
    }
    if decision.action == "dispatch":
        update["supervisor_round"] = round_number + 1
        update["intent"] = _intent_from_assignments(decision.assignments)
        update["agent_assignment_history"] = [
            *state.get("agent_assignment_history", []),
            *(item.model_dump(mode="json") for item in decision.assignments),
        ]
    elif decision.action == "finish":
        update["intent"] = state.get("intent") or "fallback"
        update["supervisor_response"] = decision.response or ""
        update["supervisor_response_source"] = "llm"
        update["sub_results"] = {
            **state.get("sub_results", {}),
            "supervisor": decision.response or "",
        }
    elif decision.action == "clarify":
        update["intent"] = state.get("intent") or ("action_pending" if active_action else "fallback")
        question = decision.clarification_question or "请补充更明确的信息。"
        update["supervisor_response"] = question
        update["supervisor_response_source"] = "llm"
        update["sub_results"] = {
            **state.get("sub_results", {}),
            "supervisor": question,
        }
    return update


def supervisor_manager_node_sync(state: ChatState) -> dict[str, Any]:
    """供 LangGraph 同步调用面的 Supervisor 适配器。"""

    return asyncio.run(supervisor_manager_node(state))


def supervisor_route(state: ChatState) -> str:
    """把严格 Supervisor action 映射为分派、治理动作或响应分支。"""

    decision = SupervisorDecision.model_validate(state.get("supervisor_decision"))
    if decision.action == "dispatch":
        return "dispatch"
    if decision.action in {"confirm_action", "reject_action"}:
        return "action"
    return "respond"


def dispatch_route(state: ChatState) -> str:
    """遇到确认/澄清结果时结束调度，否则返回 Supervisor 复核。"""

    for value in (state.get("task_results") or {}).values():
        try:
            result = AgentResult.model_validate(value)
        except (ValidationError, TypeError):
            continue
        if result.status in {"needs_confirmation", "needs_clarification"}:
            return "respond"
    return "review"


async def domain_dispatch_node(state: ChatState) -> dict[str, Any]:
    """按依赖拓扑分批执行领域任务，并隔离各任务失败。

    同一批只运行依赖已完成的任务；失败依赖会使下游显式 skipped。独立任务
    使用 gather 并行，但每个就绪批最多放行一个写能力。异常只转成对应任务
    失败，其他成功结果仍可供 Supervisor 复核和最终响应使用。
    """

    assignments = [
        AgentAssignment.model_validate(item)
        for item in (state.get("agent_assignments") or [])
    ]
    identity = identity_from_state(state)
    existing = dict(state.get("task_results") or {})
    completed = set(existing)
    pending = {item.task_id: item for item in assignments}
    batch_results: dict[str, AgentResult] = {}
    write_capabilities = get_capability_index().write_capabilities

    while pending:
        ready = [
            assignment
            for assignment in pending.values()
            if set(assignment.dependencies).issubset(completed)
        ]
        if not ready:
            for assignment in pending.values():
                batch_results[assignment.task_id] = AgentResult(
                    task_id=assignment.task_id,
                    agent=assignment.agent,
                    status="failed",
                    user_fragment="任务依赖关系无效，未执行该任务。",
                    error_code="supervisor.invalid_dependency",
                )
            break

        runnable: list[AgentAssignment] = []
        for assignment in ready:
            failed_dependencies = [
                dependency
                for dependency in assignment.dependencies
                if _result_failed(existing.get(dependency) or _dumped(batch_results.get(dependency)))
            ]
            if failed_dependencies:
                result = AgentResult(
                    task_id=assignment.task_id,
                    agent=assignment.agent,
                    status="skipped",
                    user_fragment="依赖任务未成功，本任务未执行。",
                    error_code="supervisor.dependency_failed",
                )
                batch_results[assignment.task_id] = result
                completed.add(assignment.task_id)
                pending.pop(assignment.task_id, None)
            else:
                runnable.append(assignment)

        if not runnable:
            continue
        writes = [item for item in runnable if item.capability in write_capabilities]
        selected_write_id = writes[0].task_id if writes else None
        selected = [
            item
            for item in runnable
            if item.capability not in write_capabilities or item.task_id == selected_write_id
        ]
        results = await asyncio.gather(
            *(_execute_assignment(item, state, identity) for item in selected),
            return_exceptions=True,
        )
        for assignment, value in zip(selected, results, strict=True):
            if isinstance(value, BaseException):
                error = normalize_error(value)
                result = AgentResult(
                    task_id=assignment.task_id,
                    agent=assignment.agent,
                    status="failed",
                    user_fragment="该领域服务暂时不可用，其他已成功结果仍然有效。",
                    error_code=str(error["error_code"]),
                )
                logger.warning(
                    "Sub-agent failed agent={} error_type={} error_code={}",
                    assignment.agent,
                    error["error_type"],
                    error["error_code"],
                )
            else:
                result = value
            batch_results[assignment.task_id] = result
            completed.add(assignment.task_id)
            pending.pop(assignment.task_id, None)

    ordered = {
        item.task_id: batch_results[item.task_id].model_dump(mode="json")
        for item in assignments
        if item.task_id in batch_results
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in assignments:
        result = ordered.get(item.task_id)
        if result is not None:
            grouped[item.agent].append(result)
    skill_selection = next(
        (
            result.facts.get("skill_selection")
            for result in batch_results.values()
            if result.facts.get("skill_selection")
        ),
        state.get("skill_selection"),
    )
    skill_result = state.get("skill_result")
    if skill_selection:
        selected_result = next(
            (
                result
                for result in batch_results.values()
                if result.facts.get("skill_selection") == skill_selection
            ),
            None,
        )
        if selected_result is not None:
            skill_result = {
                "status": selected_result.status,
                "facts": selected_result.facts,
                **skill_selection,
            }
    update = {
        "task_results": {**existing, **ordered},
        "domain_agent_results": {
            **state.get("domain_agent_results", {}),
            **dict(grouped),
        },
        "skill_selection": skill_selection,
        "skill_result": skill_result,
        "current_agent": "agent_dispatcher",
        "node_logs": [f"Agent dispatch completed: {len(ordered)} tasks"],
    }
    terminal_results = [
        result
        for result in batch_results.values()
        if result.status in {"needs_confirmation", "needs_clarification"}
    ]
    resumed_pending = any(
        _assignment_resumes_pending(item, state.get("pending_task"))
        for item in assignments
    )
    if terminal_results:
        fragments = [
            result.user_fragment
            for result in batch_results.values()
            if result.user_fragment
        ]
        text = "\n\n".join(fragments)
        update.update({
            "supervisor_response": text,
            "supervisor_response_source": "llm",
            "sub_results": {**state.get("sub_results", {}), "supervisor": text},
        })
        clarification_result = next(
            (
                result
                for result in batch_results.values()
                if result.status == "needs_clarification"
            ),
            None,
        )
        if clarification_result is not None and (
            state.get("pending_task") is None or resumed_pending
        ):
            source_assignment = next(
                item
                for item in assignments
                if item.task_id == clarification_result.task_id
            )
            update["pending_task"] = {
                "schema_version": "1.0",
                "task_id": source_assignment.task_id,
                "agent": source_assignment.agent,
                "capability": source_assignment.capability,
                "objective": source_assignment.objective,
                "arguments": source_assignment.arguments,
                "dependencies": list(source_assignment.dependencies),
                "clarification_question": clarification_result.user_fragment,
                "source_turn_id": state.get("turn_id") or "untracked",
            }
        elif resumed_pending:
            update["pending_task"] = None
    elif ordered and resumed_pending:
        # 已经重新执行过续接任务且不再需要澄清，清除旧任务状态。
        update["pending_task"] = None
    return update


def domain_dispatch_node_sync(state: ChatState) -> dict[str, Any]:
    """供 LangGraph 同步调用面的领域分派适配器。"""

    return asyncio.run(domain_dispatch_node(state))


async def pending_action_execution_node(state: ChatState) -> dict[str, Any]:
    """应用 Supervisor 的确认/拒绝决定并生成确定性用户结果。"""

    decision = SupervisorDecision.model_validate(state.get("supervisor_decision"))
    identity = identity_from_state(state)
    result = await resolve_pending_action(state, identity, decision.action)
    if result is None:
        text = "当前没有可确认或取消的待处理操作。"
        return {
            "intent": "action_missing",
            "supervisor_response": text,
            "supervisor_response_source": "deterministic",
            "sub_results": {**state.get("sub_results", {}), "supervisor": text},
            "current_agent": "action_governance",
            "node_logs": ["Pending action no longer exists"],
        }
    text = str((result.get("sub_results") or {}).get("supervisor") or "")
    return {
        **result,
        "pending_task": None,
        "supervisor_response": text,
        "supervisor_response_source": "deterministic",
        "current_agent": "action_governance",
        "node_logs": ["Pending action decision applied"],
    }


def pending_action_execution_node_sync(state: ChatState) -> dict[str, Any]:
    """供 LangGraph 同步调用面的治理动作适配器。"""

    return asyncio.run(pending_action_execution_node(state))


async def _execute_assignment(
    assignment: AgentAssignment,
    state: ChatState,
    identity: RequestIdentityContext,
) -> AgentResult:
    """选择与 assignment.agent 固定对应的子图并校验其结果。"""

    graph = {
        "knowledge_agent": knowledge_agent_graph,
        "telecom_agent": telecom_agent_graph,
        "retail_agent": retail_agent_graph,
    }[assignment.agent]
    output = await graph.ainvoke({
        "parent_state": state,
        "identity": identity,
        "assignment": assignment.model_dump(mode="json"),
        "result": None,
    })
    return AgentResult.model_validate(output["result"])


async def _active_action(identity: RequestIdentityContext) -> dict[str, Any] | None:
    """读取并投影活跃动作，只暴露意图判断需要的非敏感字段。"""

    try:
        actions = get_action_service()
    except RuntimeError:
        return None
    active = await asyncio.to_thread(actions.get_active, identity)
    if active is None:
        return None
    return {
        "action_id": active.action_id,
        "tool_name": active.tool_name,
        "status": active.status,
        "impact_summary": active.impact_summary,
        "expires_at": active.expires_at.isoformat(),
    }


def _intent_from_assignments(assignments: tuple[AgentAssignment, ...]) -> str:
    """从 Agent/能力归属汇总本轮电信、零售或复合意图标签。"""

    domains = set()
    for item in assignments:
        if item.agent == "telecom_agent" or item.capability == "telecom_troubleshooting":
            domains.add("telecom")
        elif item.agent == "retail_agent" or item.capability == "retail_policy":
            domains.add("retail")
    if len(domains) > 1:
        return "composite"
    return next(iter(domains), "fallback")


def _result_failed(value: Any) -> bool:
    """判断依赖结果是否失败；畸形结果按失败关闭处理。"""

    if value is None:
        return False
    try:
        return AgentResult.model_validate(value).status in {"failed", "skipped"}
    except (ValidationError, TypeError):
        return True


def _assignment_resumes_pending(
    assignment: AgentAssignment,
    pending_task: Any,
) -> bool:
    """按 Agent 与能力识别续接任务，避免无关查询误清理未完成状态。"""

    if not isinstance(pending_task, dict):
        return False
    return (
        pending_task.get("agent") == assignment.agent
        and pending_task.get("capability") == assignment.capability
    )


def _dumped(result: AgentResult | None) -> dict[str, Any] | None:
    """把可选任务结果转换为可放入图状态的 JSON 数据。"""

    return result.model_dump(mode="json") if result is not None else None


def identity_from_state(state: ChatState) -> RequestIdentityContext:
    """从应用预填的状态恢复可信身份上下文，不读取模型生成字段。"""

    return RequestIdentityContext(
        user_id=state.get("user_id") or "anonymous",
        session_id=state["session_id"],
        turn_id=state.get("turn_id") or "untracked",
        request_id=state.get("request_id"),
        identity_source=state.get("identity_source") or "request_body",
        auth_strength=state.get("auth_strength") or "unverified_frontend",
    )
