from __future__ import annotations

import re

from domain.customer_service_agent.orchestration.models import RouteDecision, TaskPlan, TaskSpec
from domain.customer_service_agent.orchestration.router import WRITE_CAPABILITIES


def build_task_plan(query: str, decision: RouteDecision) -> TaskPlan:
    tasks = []
    for index, capability in enumerate(decision.capabilities):
        domain = (
            "telecom"
            if capability.startswith(("plan_", "data_", "roaming", "current_", "usage", "telecom_"))
            else "retail" if capability != "fallback" else "shared"
        )
        tasks.append(TaskSpec(
            task_id=f"T{index + 1}",
            domain=domain,
            capability=capability,
            effect="write" if capability in WRITE_CAPABILITIES else "read",
            arguments=extract_entities(query),
        ))
    return TaskPlan(tasks=tuple(tasks))


def extract_entities(query: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("line_id", "plan_id", "order_id", "address_id", "payment_method_id"):
        match = re.search(rf"{key}\s*[:=：]\s*([A-Za-z0-9_-]{{1,64}})", query, re.I)
        if match:
            result[key] = match.group(1)
    return result
