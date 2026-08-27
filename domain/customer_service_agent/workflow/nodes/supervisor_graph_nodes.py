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
from domain.customer_service_agent.orchestration.pending_action_resolver import resolve_pending_action
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.identity import RequestIdentityContext
from pkg.log.logger import get_logger
from pkg.telemetry import normalize_error

logger = get_logger()


async def supervisor_manager_node(state: ChatState) -> dict[str, Any]:
    identity = identity_from_state(state)
    active_action = await _active_action(identity)
    round_number = int(state.get("supervisor_round") or 0)
    decision = await decide_next_step(
        state,
        active_action=active_action,
        allow_dispatch=round_number < MAX_SUPERVISOR_ROUNDS,
    )
    update: dict[str, Any] = {
        "normalized_query": decision.standalone_query,
        "supervisor_decision": decision.model_dump(mode="json"),
        "active_action": active_action,
        "agent_assignments": [item.model_dump(mode="json") for item in decision.assignments],
        "current_agent": "supervisor",
        "node_logs": [f"Supervisor decision: {decision.action}"],
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
    return asyncio.run(supervisor_manager_node(state))


def supervisor_route(state: ChatState) -> str:
    decision = SupervisorDecision.model_validate(state.get("supervisor_decision"))
    if decision.action == "dispatch":
        return "dispatch"
    if decision.action in {"confirm_action", "reject_action"}:
        return "action"
    return "respond"


def dispatch_route(state: ChatState) -> str:
    for value in (state.get("task_results") or {}).values():
        try:
            result = AgentResult.model_validate(value)
        except (ValidationError, TypeError):
            continue
        if result.status in {"needs_confirmation", "needs_clarification"}:
            return "respond"
    return "review"


async def domain_dispatch_node(state: ChatState) -> dict[str, Any]:
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
        selected = [writes[0]] if writes else runnable
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
    return update


def domain_dispatch_node_sync(state: ChatState) -> dict[str, Any]:
    return asyncio.run(domain_dispatch_node(state))


async def pending_action_execution_node(state: ChatState) -> dict[str, Any]:
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
        "supervisor_response": text,
        "supervisor_response_source": "deterministic",
        "current_agent": "action_governance",
        "node_logs": ["Pending action decision applied"],
    }


def pending_action_execution_node_sync(state: ChatState) -> dict[str, Any]:
    return asyncio.run(pending_action_execution_node(state))


def result_aggregator_node(state: ChatState) -> dict[str, Any]:
    """Compatibility helper for callers that need deterministic result ordering."""
    fragments: list[str] = []
    for value in (state.get("task_results") or {}).values():
        try:
            result = AgentResult.model_validate(value)
        except (ValidationError, TypeError):
            continue
        if result.user_fragment:
            fragments.append(result.user_fragment)
    text = "\n\n".join(fragments) or "请补充更明确的业务问题。"
    return {
        "sub_results": {**state.get("sub_results", {}), "supervisor": text},
        "supervisor_response": text,
        "supervisor_response_source": "deterministic",
        "current_agent": "result_aggregator",
        "node_logs": [f"Aggregated {len(fragments)} agent results"],
    }


async def _execute_assignment(
    assignment: AgentAssignment,
    state: ChatState,
    identity: RequestIdentityContext,
) -> AgentResult:
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
    if value is None:
        return False
    try:
        return AgentResult.model_validate(value).status in {"failed", "skipped"}
    except (ValidationError, TypeError):
        return True


def _dumped(result: AgentResult | None) -> dict[str, Any] | None:
    return result.model_dump(mode="json") if result is not None else None


def identity_from_state(state: ChatState) -> RequestIdentityContext:
    return RequestIdentityContext(
        user_id=state.get("user_id") or "anonymous",
        session_id=state["session_id"],
        turn_id=state.get("turn_id") or "untracked",
        request_id=state.get("request_id"),
        identity_source=state.get("identity_source") or "request_body",
        auth_strength=state.get("auth_strength") or "unverified_frontend",
    )
