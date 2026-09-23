"""Supervisor 与领域 Agent 之间传递的严格结构化协议。"""

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
    """一个分派任务的标准结果。

    `facts` 供后续 Agent/Supervisor 继续推理，`user_fragment` 供最终响应合成；
    两者分离可避免把内部结构直接暴露给用户。
    """

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
    """仅由 Supervisor 产生的类型化工作单元。

    dependencies 引用本轮或历史 task_id，由确定性编排器检查存在性和环路。
    arguments 只是候选输入，领域 Agent 仍须执行工具白名单与 Schema 校验。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_id: str = Field(min_length=1, max_length=64)
    agent: AgentName
    objective: str = Field(min_length=1, max_length=4000)
    capability: str = Field(min_length=1, max_length=128)
    dependencies: tuple[str, ...] = ()
    arguments: dict[str, Any] = Field(default_factory=dict)


class PendingWritePlan(BaseModel):
    """Ordered write intents with exactly one item eligible for proposal at a time.

    The plan stores intents rather than frozen tool calls.  Each item is sent back
    through its domain Agent when it becomes current so ownership, current state,
    resource version, and write arguments are read and validated again.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: str = "1.0"
    plan_id: str = Field(min_length=1, max_length=64)
    source_turn_id: str = Field(min_length=1, max_length=64)
    assignments: tuple[AgentAssignment, ...] = Field(min_length=2, max_length=8)
    current_index: int = Field(default=0, ge=0)
    active_action_id: str | None = Field(default=None, max_length=64)
    completed_action_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)

    @model_validator(mode="after")
    def validate_cursor(self) -> "PendingWritePlan":
        if self.current_index >= len(self.assignments):
            raise ValueError("write plan cursor is outside the assignment list")
        task_ids = [item.task_id for item in self.assignments]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("write plan task ids must be unique")
        return self

    @property
    def current_assignment(self) -> AgentAssignment:
        return self.assignments[self.current_index]

    @property
    def remaining_count(self) -> int:
        return len(self.assignments) - self.current_index - 1


class SupervisorDecision(BaseModel):
    """每轮有界 Supervisor 调度的严格输出契约。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: SupervisorAction
    standalone_query: str = Field(min_length=1, max_length=4000)
    assignments: tuple[AgentAssignment, ...] = Field(default_factory=tuple, max_length=8)
    response: str | None = Field(default=None, max_length=20_000)
    clarification_question: str | None = Field(default=None, max_length=4000)
    confidence: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_action_payload(self) -> "SupervisorDecision":
        """保证不同 action 只携带与其语义匹配的字段组合。"""

        if self.action == "dispatch" and not self.assignments:
            raise ValueError("dispatch requires assignments")
        if self.action != "dispatch" and self.assignments:
            raise ValueError("only dispatch may contain assignments")
        if self.action == "finish" and not self.response:
            raise ValueError("finish requires response")
        if self.action == "clarify" and not self.clarification_question:
            raise ValueError("clarify requires clarification_question")
        return self


class ReadToolCall(BaseModel):
    """并行批次中一个可独立执行、无副作用的读取调用。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tool_name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentStepDecision(BaseModel):
    """领域 Agent 的一个有界步骤；write 只表示提议，绝不表示已执行。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["tool_call", "tool_calls", "propose_write", "final", "clarify"]
    tool_name: str | None = Field(default=None, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_calls: tuple[ReadToolCall, ...] = Field(default_factory=tuple, max_length=3)
    response: str | None = Field(default=None, max_length=20_000)
    impact_summary: str | None = Field(default=None, max_length=2000)
    missing_fields: tuple[str, ...] = Field(default_factory=tuple, max_length=32)

    @model_validator(mode="after")
    def validate_step_payload(self) -> "AgentStepDecision":
        """拒绝缺少工具/响应或混合单调用与批调用的非法步骤。"""

        if self.action in {"tool_call", "propose_write"} and not self.tool_name:
            raise ValueError("tool action requires tool_name")
        if self.action == "tool_calls" and not self.tool_calls:
            raise ValueError("tool_calls requires at least one read call")
        if self.action != "tool_calls" and self.tool_calls:
            raise ValueError("only tool_calls may contain a read batch")
        if self.action == "tool_calls" and self.tool_name:
            raise ValueError("tool_calls cannot contain a single tool_name")
        if self.action in {"final", "clarify"} and not self.response:
            raise ValueError("response action requires response")
        if self.action == "clarify" and not self.missing_fields:
            raise ValueError("clarify requires missing_fields")
        if self.action != "clarify" and self.missing_fields:
            raise ValueError("only clarify may contain missing_fields")
        if self.action == "propose_write" and not self.impact_summary:
            raise ValueError("write proposal requires impact_summary")
        return self
