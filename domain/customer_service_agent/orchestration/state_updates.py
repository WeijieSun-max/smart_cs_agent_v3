from __future__ import annotations

from typing import Any

from domain.customer_service_agent.orchestration.models import RouteDecision, TaskPlan
from domain.customer_service_agent.workflow.entity.chat_state import ChatState


def build_supervisor_result(
    state: ChatState,
    agent: str,
    text: str,
    *,
    route: RouteDecision | None = None,
    plan: TaskPlan | None = None,
    task_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "intent": agent,
        "current_agent": "supervisor",
        "sub_results": {**state.get("sub_results", {}), "supervisor": text},
        "route_decision": route.model_dump(mode="json") if route else None,
        "task_plan": plan.model_dump(mode="json") if plan else None,
        "task_results": task_results or {},
        "node_logs": [f"Supervisor completed: {agent}"],
    }
