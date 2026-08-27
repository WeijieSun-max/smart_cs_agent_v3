"""Typed legacy plan adapter without text parsing.

The manager workflow uses ``AgentAssignment`` directly. This adapter remains for
bounded-read utilities and accepts only entities already produced by an LLM.
"""

from __future__ import annotations

from typing import Any

from domain.customer_service_agent.orchestration.capability_index import get_capability_index
from domain.customer_service_agent.orchestration.models import RouteDecision, TaskPlan, TaskSpec

_REACT_CAPABILITIES = frozenset({"current_plan", "usage", "product_query"})


def build_task_plan(
    query: str,
    decision: RouteDecision,
    *,
    entities: dict[str, Any] | None = None,
    requires_planning: bool = False,
) -> TaskPlan:
    del query
    index = get_capability_index()
    tasks = []
    arguments = dict(entities or {})
    for position, capability in enumerate(decision.capabilities, start=1):
        domain = _capability_domain(capability)
        tasks.append(TaskSpec(
            task_id=f"T{position}",
            domain=domain,
            capability=capability,
            effect="write" if capability in index.write_capabilities else "read",
            execution_mode=(
                "react"
                if requires_planning and capability in _REACT_CAPABILITIES
                else "direct"
            ),
            arguments=arguments,
        ))
    return TaskPlan(tasks=tuple(tasks))


def extract_entities(_query: str) -> dict[str, Any]:
    """Text entity extraction was removed; use structured LLM arguments."""
    return {}


def _capability_domain(capability: str) -> str:
    if capability == "fallback":
        return "shared"
    for tool in get_capability_index().tools:
        if capability in tool.capabilities and tool.domain in {"telecom", "retail"}:
            return tool.domain
    if capability == "telecom_troubleshooting":
        return "knowledge"
    if capability == "retail_policy":
        return "knowledge"
    return "shared"
