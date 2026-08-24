from __future__ import annotations

import asyncio
from typing import Any

from domain.customer_service_agent.orchestration import (
    build_task_plan,
    execute_composite,
    execute_read_task,
    route_request,
)
from domain.customer_service_agent.orchestration.pending_action_resolver import resolve_pending_action
from domain.customer_service_agent.orchestration.skill_executor import execute_plan_recommendation
from domain.customer_service_agent.orchestration.state_updates import build_supervisor_result
from domain.customer_service_agent.orchestration.write_proposal import prepare_write_proposal
from domain.customer_service_agent.retrieval import retrieve_grounded_answer
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.identity import RequestIdentityContext


async def supervisor_node(state: ChatState) -> dict[str, Any]:
    query = (state.get("raw_query") or "").strip()
    identity = _identity(state)
    pending = await resolve_pending_action(state, identity)
    if pending is not None:
        return pending

    decision = route_request(query)
    if "plan_recommendation" in decision.capabilities:
        return await execute_plan_recommendation(state, identity)

    plan = build_task_plan(query, decision)
    plan.validate_dag()
    if decision.composite:
        results = await execute_composite(plan, state, identity)
        text = "\n\n".join(result.user_fragment for result in results if result.user_fragment)
        return build_supervisor_result(
            state,
            "composite",
            text,
            route=decision,
            plan=plan,
            task_results={item.task_id: item.model_dump(mode="json") for item in results},
        )

    capability = decision.capabilities[0] if decision.capabilities else "fallback"
    if capability in {"telecom_troubleshooting", "retail_policy"}:
        return await retrieve_grounded_answer(state, decision.domains[0], capability)
    if capability in {"current_plan", "usage", "order_query", "product_query"}:
        result = await execute_read_task(plan.tasks[0], identity)
        return build_supervisor_result(
            state,
            decision.domains[0],
            result.user_fragment,
            route=decision,
            plan=plan,
            task_results={result.task_id: result.model_dump(mode="json")},
        )
    proposal = await prepare_write_proposal(query, capability, identity)
    if proposal is not None:
        return build_supervisor_result(
            state,
            decision.domains[0],
            proposal,
            route=decision,
            plan=plan,
        )
    return build_supervisor_result(
        state,
        "fallback",
        "请说明要查询或办理的具体业务。可处理套餐/流量/漫游、通信故障指导，以及商品、订单、取消、修改、退换货和退款。",
        route=decision,
        plan=plan,
    )


def supervisor_node_sync(state: ChatState) -> dict[str, Any]:
    return asyncio.run(supervisor_node(state))


def _identity(state: ChatState) -> RequestIdentityContext:
    return RequestIdentityContext(
        user_id=state.get("user_id") or "anonymous",
        session_id=state["session_id"],
        turn_id=state.get("turn_id") or "untracked",
        request_id=state.get("request_id"),
        identity_source=state.get("identity_source") or "request_body",
        auth_strength=state.get("auth_strength") or "unverified_frontend",
    )
