from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AgentName = Literal["knowledge_agent", "telecom_agent", "retail_agent"]
SupervisorAction = Literal[
    "dispatch",
    "finish",
    "clarify",
    "confirm_action",
    "reject_action",
]


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
    domain: Literal["knowledge", "telecom", "retail", "shared"]
    capability: str
    dependencies: tuple[str, ...] = ()
    effect: Literal["read", "write"] = "read"
    execution_mode: Literal["direct", "react"] = "direct"
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


class QueryUnderstandingResult(BaseModel):
    schema_version: str = "1.0"
    standalone_query: str
    domains: tuple[Literal["telecom", "retail", "fallback"], ...]
    capabilities: tuple[str, ...]
    entities: dict[str, Any] = Field(default_factory=dict)
    temporal_range: dict[str, str] | None = None
    ambiguity: bool = False
    missing_fields: tuple[str, ...] = ()
    requires_planning: bool = False
    confidence: float = Field(ge=0, le=1)
    source: Literal["llm", "fallback"] = "llm"


class OrderResolution(BaseModel):
    status: Literal["resolved", "multiple", "not_found"]
    order_id: str | None = None
    candidates: tuple[dict[str, Any], ...] = ()
    user_fragment: str = ""


class AgentResult(BaseModel):
    schema_version: str = "1.0"
    task_id: str
    status: Literal[
        "succeeded",
        "failed",
        "needs_confirmation",
        "needs_clarification",
        "skipped",
    ]
    agent: AgentName | None = None
    facts: dict[str, Any] = Field(default_factory=dict)
    user_fragment: str = ""
    tool_receipts: tuple[ToolReceipt, ...] = ()
    pending_action_id: str | None = None
    error_code: str | None = None


class AgentAssignment(BaseModel):
    """A typed unit of work emitted only by the LLM supervisor."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_id: str = Field(min_length=1, max_length=64)
    agent: AgentName
    objective: str = Field(min_length=1, max_length=4000)
    capability: str = Field(min_length=1, max_length=128)
    dependencies: tuple[str, ...] = ()
    arguments: dict[str, Any] = Field(default_factory=dict)


class SupervisorDecision(BaseModel):
    """Strict output contract for each bounded manager round."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: SupervisorAction
    standalone_query: str = Field(min_length=1, max_length=4000)
    assignments: tuple[AgentAssignment, ...] = Field(default_factory=tuple, max_length=8)
    response: str | None = Field(default=None, max_length=20_000)
    clarification_question: str | None = Field(default=None, max_length=4000)
    confidence: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_action_payload(self) -> "SupervisorDecision":
        if self.action == "dispatch" and not self.assignments:
            raise ValueError("dispatch requires assignments")
        if self.action != "dispatch" and self.assignments:
            raise ValueError("only dispatch may contain assignments")
        if self.action == "finish" and not self.response:
            raise ValueError("finish requires response")
        if self.action == "clarify" and not self.clarification_question:
            raise ValueError("clarify requires clarification_question")
        return self


class AgentStepDecision(BaseModel):
    """One bounded domain-agent step; write means proposal, never execution."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["tool_call", "propose_write", "final", "clarify"]
    tool_name: str | None = Field(default=None, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    response: str | None = Field(default=None, max_length=20_000)
    impact_summary: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_step_payload(self) -> "AgentStepDecision":
        if self.action in {"tool_call", "propose_write"} and not self.tool_name:
            raise ValueError("tool action requires tool_name")
        if self.action in {"final", "clarify"} and not self.response:
            raise ValueError("response action requires response")
        if self.action == "propose_write" and not self.impact_summary:
            raise ValueError("write proposal requires impact_summary")
        return self
