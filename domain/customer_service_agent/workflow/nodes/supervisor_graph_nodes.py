from __future__ import annotations

import asyncio
from typing import Any

from domain.customer_service_agent.orchestration.domain_agents import retail_agent_graph, telecom_agent_graph
from domain.customer_service_agent.orchestration.models import AgentResult, QueryUnderstandingResult, RouteDecision, TaskPlan
from domain.customer_service_agent.orchestration.pending_action_resolver import resolve_pending_action
from domain.customer_service_agent.orchestration.planner import build_task_plan
from domain.customer_service_agent.orchestration.query_understanding import route_from_understanding, understand_query
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.identity import RequestIdentityContext
from pkg.log.logger import get_logger
from pkg.telemetry import normalize_error

logger = get_logger()
_FALLBACK_TEXT = "请说明要查询或办理的具体业务。可处理套餐/流量/漫游、通信故障指导，以及商品、订单、取消、修改、退换货和退款。"


async def pending_action_gate_node(state: ChatState) -> dict[str, Any]:
    pending = await resolve_pending_action(state, identity_from_state(state))
    if pending is None:
        return {
            "pending_action_handled": False,
            "current_agent": "pending_action_gate",
            "node_logs": ["当前会话无待确认操作"],
        }
    return {
        **pending,
        "pending_action_handled": True,
        "node_logs": ["待确认操作已优先处理"],
    }


def pending_action_gate_node_sync(state: ChatState) -> dict[str, Any]:
    return asyncio.run(pending_action_gate_node(state))


def pending_action_route(state: ChatState) -> str:
    return "handled" if state.get("pending_action_handled") else "continue"


def supervisor_router_node(state: ChatState) -> dict[str, Any]:
    data = state.get("query_understanding")
    understanding = QueryUnderstandingResult.model_validate(data) if data else understand_query(state)
    decision = route_from_understanding({**state, "query_understanding": understanding.model_dump(mode="json")})
    return {
        "normalized_query": understanding.standalone_query,
        "query_understanding": understanding.model_dump(mode="json"),
        "route_decision": decision.model_dump(mode="json"),
        "current_agent": "supervisor_router",
        "node_logs": [f"Supervisor routed: {','.join(decision.domains)}"],
    }


def supervisor_planner_node(state: ChatState) -> dict[str, Any]:
    decision = RouteDecision.model_validate(state["route_decision"])
    understanding = QueryUnderstandingResult.model_validate(state["query_understanding"])
    plan = build_task_plan(
        understanding.standalone_query,
        decision,
        entities=understanding.entities,
        requires_planning=understanding.requires_planning,
    )
    plan.validate_dag()
    return {
        "task_plan": plan.model_dump(mode="json"),
        "current_agent": "supervisor_planner",
        "node_logs": [f"Supervisor planned: {len(plan.tasks)} tasks"],
    }


async def domain_dispatch_node(state: ChatState) -> dict[str, Any]:
    plan = TaskPlan.model_validate(state["task_plan"])
    identity = identity_from_state(state)
    tasks_by_domain = {
        "telecom": [task.model_dump(mode="json") for task in plan.tasks if task.domain == "telecom"],
        "retail": [task.model_dump(mode="json") for task in plan.tasks if task.domain == "retail"],
    }
    parent_state = {**state, "raw_query": state.get("normalized_query") or state["raw_query"]}
    calls = []
    domains = []
    if tasks_by_domain["telecom"]:
        domains.append("telecom")
        calls.append(telecom_agent_graph.ainvoke({
            "parent_state": parent_state,
            "identity": identity,
            "tasks": tasks_by_domain["telecom"],
            "results": [],
        }))
    if tasks_by_domain["retail"]:
        domains.append("retail")
        calls.append(retail_agent_graph.ainvoke({
            "parent_state": parent_state,
            "identity": identity,
            "tasks": tasks_by_domain["retail"],
            "results": [],
        }))
    outputs = await asyncio.gather(*calls, return_exceptions=True)
    domain_results: dict[str, list[dict[str, Any]]] = {}
    for domain, output in zip(domains, outputs, strict=True):
        if isinstance(output, BaseException):
            error = normalize_error(output)
            logger.warning(
                "Domain agent failed domain={} error_type={} error_code={}",
                domain,
                error["error_type"],
                error["error_code"],
            )
            safe_message = getattr(
                output,
                "safe_message",
                f"{domain} 领域服务暂时不可用，其他已成功结果仍然有效。",
            )
            domain_results[domain] = [
                AgentResult(
                    task_id=task["task_id"],
                    status="failed",
                    user_fragment=safe_message,
                    error_code=str(error["error_code"]),
                ).model_dump(mode="json")
                for task in tasks_by_domain[domain]
            ]
        else:
            domain_results[domain] = list(output.get("results", []))
    shared = [task for task in plan.tasks if task.domain == "shared"]
    if shared:
        domain_results["shared"] = [
            AgentResult(
                task_id=task.task_id,
                status="succeeded",
                user_fragment=_FALLBACK_TEXT,
            ).model_dump(mode="json")
            for task in shared
        ]
    return {
        "domain_agent_results": domain_results,
        "current_agent": "domain_dispatch",
        "node_logs": [f"Domain dispatch completed: {','.join(domain_results) or 'none'}"],
    }


def domain_dispatch_node_sync(state: ChatState) -> dict[str, Any]:
    return asyncio.run(domain_dispatch_node(state))


def result_aggregator_node(state: ChatState) -> dict[str, Any]:
    plan = TaskPlan.model_validate(state["task_plan"])
    raw_results = state.get("domain_agent_results") or {}
    by_task_id = {
        result.task_id: result
        for values in raw_results.values()
        for result in (AgentResult.model_validate(item) for item in values)
    }
    ordered = [by_task_id[task.task_id] for task in plan.tasks if task.task_id in by_task_id]
    fragments = [result.user_fragment for result in ordered if result.user_fragment]
    text = "\n\n".join(fragments) if fragments else _FALLBACK_TEXT
    route = RouteDecision.model_validate(state["route_decision"])
    intent = "composite" if route.composite else route.domains[0]
    skill_selection = next(
        (result.facts.get("skill_selection") for result in ordered if result.facts.get("skill_selection")),
        None,
    )
    skill_result = next(
        (result.facts.get("skill_result") for result in ordered if result.facts.get("skill_result")),
        None,
    )
    return {
        "intent": intent,
        "sub_results": {**state.get("sub_results", {}), "supervisor": text},
        "task_results": {result.task_id: result.model_dump(mode="json") for result in ordered},
        "skill_selection": skill_selection,
        "skill_result": skill_result,
        "current_agent": "result_aggregator",
        "node_logs": [f"Aggregated {len(ordered)} domain task results"],
    }


def identity_from_state(state: ChatState) -> RequestIdentityContext:
    return RequestIdentityContext(
        user_id=state.get("user_id") or "anonymous",
        session_id=state["session_id"],
        turn_id=state.get("turn_id") or "untracked",
        request_id=state.get("request_id"),
        identity_source=state.get("identity_source") or "request_body",
        auth_strength=state.get("auth_strength") or "unverified_frontend",
    )
