from __future__ import annotations

import asyncio
from typing import Any

from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.customer_service_agent.workflow.nodes.supervisor_graph_nodes import (
    domain_dispatch_node,
    dispatch_route,
    pending_action_execution_node,
    supervisor_manager_node,
    supervisor_route,
)


async def supervisor_node(state: ChatState) -> dict[str, Any]:
    """Run the same bounded manager loop for direct service/test callers."""
    working: ChatState = {**state}
    while True:
        working.update(await supervisor_manager_node(working))
        route = supervisor_route(working)
        if route == "dispatch":
            working.update(await domain_dispatch_node(working))
            if dispatch_route(working) == "respond":
                return {**working, "current_agent": "supervisor"}
            continue
        if route == "action":
            working.update(await pending_action_execution_node(working))
        return {**working, "current_agent": "supervisor"}


def supervisor_node_sync(state: ChatState) -> dict[str, Any]:
    return asyncio.run(supervisor_node(state))
