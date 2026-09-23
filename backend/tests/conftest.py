from __future__ import annotations

import os
from typing import Any

import pytest


# Encryption is mandatory outside debug mode. Tests use an explicit stable key
# so they do not depend on a developer's ignored .env file.
os.environ.setdefault("PII_ENCRYPTION_KEY", "smart-cs-agent-test-only-encryption-key")


@pytest.fixture
def run_supervisor():
    """Run the bounded manager loop without compiling the full workflow graph."""
    from domain.customer_service_agent.workflow.nodes.supervisor_graph_nodes import (
        dispatch_route,
        domain_dispatch_node,
        pending_action_execution_node,
        supervisor_manager_node,
        supervisor_route,
    )

    async def run(state: dict[str, Any]) -> dict[str, Any]:
        working = {**state}
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

    return run
