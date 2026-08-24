from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolReceipt(BaseModel):
    schema_version: str = "1.0"
    action_id: str
    idempotency_key: str
    tool_name: str
    status: Literal["succeeded", "failed", "indeterminate"]
    resource_type: str
    resource_id: str
    version_before: int | None = None
    version_after: int | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class RouteDecision(BaseModel):
    schema_version: str = "1.0"
    domains: tuple[Literal["telecom", "retail", "fallback"], ...]
    capabilities: tuple[str, ...]
    confidence: float = Field(ge=0, le=1)
    composite: bool = False
    risk_level: Literal["low", "medium", "high"] = "low"


class TaskSpec(BaseModel):
    schema_version: str = "1.0"
    task_id: str
    domain: Literal["telecom", "retail", "shared"]
    capability: str
    dependencies: tuple[str, ...] = ()
    effect: Literal["read", "write"] = "read"
    arguments: dict[str, Any] = Field(default_factory=dict)


class TaskPlan(BaseModel):
    schema_version: str = "1.0"
    tasks: tuple[TaskSpec, ...]

    def validate_dag(self) -> None:
        ids = {task.task_id for task in self.tasks}
        if len(ids) != len(self.tasks):
            raise ValueError("duplicate task id")
        if any(set(task.dependencies) - ids for task in self.tasks):
            raise ValueError("unknown task dependency")
        visiting: set[str] = set()
        visited: set[str] = set()
        graph = {task.task_id: task.dependencies for task in self.tasks}

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("task plan contains a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in graph[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in graph:
            visit(task_id)


class AgentResult(BaseModel):
    schema_version: str = "1.0"
    task_id: str
    status: Literal["succeeded", "failed", "needs_confirmation", "skipped"]
    facts: dict[str, Any] = Field(default_factory=dict)
    user_fragment: str = ""
    tool_receipts: tuple[ToolReceipt, ...] = ()
    pending_action_id: str | None = None
    error_code: str | None = None
