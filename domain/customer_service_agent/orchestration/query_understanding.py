"""Compatibility projection of the LLM supervisor decision.

The active workflow no longer has a separate query-understanding classifier.
This module contains no keyword or regular-expression routing.
"""

from __future__ import annotations

import asyncio
from typing import Any

from domain.customer_service_agent.agents.supervisor_agent import decide_next_step
from domain.customer_service_agent.orchestration.capability_index import get_capability_index
from domain.customer_service_agent.orchestration.models import QueryUnderstandingResult, RouteDecision
from domain.customer_service_agent.workflow.entity.chat_state import ChatState

_CAPABILITIES = get_capability_index().routable_capabilities()


def query_understanding_node(state: ChatState) -> dict[str, Any]:
    result = understand_query(state)
    return {
        "normalized_query": result.standalone_query,
        "query_understanding": result.model_dump(mode="json"),
        "current_agent": "supervisor",
        "node_logs": ["查询理解由 LLM Supervisor 完成"],
    }


def understand_query(state: ChatState) -> QueryUnderstandingResult:
    decision = asyncio.run(
        decide_next_step(state, active_action=None, allow_dispatch=True)
    )
    capabilities = tuple(item.capability for item in decision.assignments) or ("fallback",)
    domains = tuple(dict.fromkeys(_assignment_domain(item.agent, item.capability) for item in decision.assignments))
    if not domains:
        domains = ("fallback",)
    entities: dict[str, Any] = {}
    for assignment in decision.assignments:
        entities.update(assignment.arguments)
    temporal = _temporal_range(entities)
    return QueryUnderstandingResult(
        standalone_query=decision.standalone_query,
        domains=domains,
        capabilities=capabilities,
        entities=entities,
        temporal_range=temporal,
        ambiguity=decision.action == "clarify",
        requires_planning=len(decision.assignments) > 1,
        confidence=decision.confidence,
        source="llm",
    )


def route_from_understanding(state: ChatState) -> RouteDecision:
    result = QueryUnderstandingResult.model_validate(state.get("query_understanding"))
    capabilities = result.capabilities or ("fallback",)
    domains = result.domains or ("fallback",)
    return RouteDecision(
        domains=domains,
        capabilities=capabilities,
        confidence=result.confidence,
        composite=len(domains) > 1 or len(capabilities) > 1,
        risk_level=(
            "medium"
            if any(item in get_capability_index().write_capabilities for item in capabilities)
            else "low"
        ),
    )


def _assignment_domain(agent: str, capability: str) -> str:
    if agent == "telecom_agent" or capability == "telecom_troubleshooting":
        return "telecom"
    if agent == "retail_agent" or capability == "retail_policy":
        return "retail"
    return "fallback"


def _temporal_range(entities: dict[str, Any]) -> dict[str, str] | None:
    start = entities.get("start_date")
    end = entities.get("end_date")
    if isinstance(start, str) and isinstance(end, str):
        return {"start": start, "end": end}
    return None
