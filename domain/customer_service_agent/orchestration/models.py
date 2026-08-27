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
