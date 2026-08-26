from __future__ import annotations

import asyncio
from typing import Any

from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.customer_service_agent.workflow.nodes.supervisor_graph_nodes import (
    domain_dispatch_node,
    pending_action_gate_node,
    result_aggregator_node,
    supervisor_planner_node,
    supervisor_router_node,
)


async def supervisor_node(state: ChatState) -> dict[str, Any]:
    pending = await pending_action_gate_node(state)
    if pending.get("pending_action_handled"):
        return pending
    working = {**state, **pending}
    routed = supervisor_router_node(working)
    working.update(routed)
    planned = supervisor_planner_node(working)
    working.update(planned)
    dispatched = await domain_dispatch_node(working)
    working.update(dispatched)
    aggregated = result_aggregator_node(working)
    return {**routed, **planned, **dispatched, **aggregated, "current_agent": "supervisor"}


def supervisor_node_sync(state: ChatState) -> dict[str, Any]:
    return asyncio.run(supervisor_node(state))
