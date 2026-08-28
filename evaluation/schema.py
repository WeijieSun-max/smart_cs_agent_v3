from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RewardComponent(StrEnum):
    STATE = "STATE"
    SIDE_EFFECT = "SIDE_EFFECT"
    GOVERNANCE = "GOVERNANCE"
    COMMUNICATE = "COMMUNICATE"
    GROUNDING = "GROUNDING"
    NL_ASSERTION = "NL_ASSERTION"
    SAFETY = "SAFETY"
    ACTION = "ACTION"


class StateAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    scope: Literal["environment", "final_state"] = "environment"
    path: str = Field(min_length=1, max_length=512)
    op: Literal[
        "eq",
        "ne",
        "exists",
        "not_exists",
        "contains",
        "in",
        "count_eq",
    ] = "eq"
    expected: Any = None


class SideEffectAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    kind: Literal[
        "count",
        "unchanged",
        "unchanged_for_other_users",
        "exactly_one_receipt",
    ]
    target: str | None = None
    where: dict[str, Any] = Field(default_factory=dict)
    expected: int | None = None
    user_id: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "SideEffectAssertion":
        if self.kind == "count" and (not self.target or self.expected is None):
            raise ValueError("count side-effect assertions require target and expected")
        if self.kind == "unchanged" and not self.target:
            raise ValueError("unchanged side-effect assertions require target")
        return self


class ForbiddenCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: tuple[str, ...] = ()
    behaviors: tuple[str, ...] = ()
    response_terms: tuple[str, ...] = ()


class EvaluationBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_turns: int | None = Field(default=None, ge=1, le=100)
    max_supervisor_rounds_per_turn: int | None = Field(default=None, ge=1, le=20)
    max_llm_calls: int | None = Field(default=None, ge=0)
    max_tool_calls: int | None = Field(default=None, ge=0)
    max_elapsed_ms: float | None = Field(default=None, gt=0)


class EvaluationCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state_assertions: tuple[StateAssertion, ...] = ()
    side_effect_assertions: tuple[SideEffectAssertion, ...] = ()
    process_assertions: tuple[str, ...] = ()
    communicate: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    nl_assertions: tuple[str, ...] = ()
    forbidden: ForbiddenCriteria = Field(default_factory=ForbiddenCriteria)
    vetoes: tuple[str, ...] = ()
    budgets: EvaluationBudgets = Field(default_factory=EvaluationBudgets)
    reward_basis: tuple[RewardComponent, ...] = (RewardComponent.SAFETY,)
    external_assertions: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def validate_reward_basis(self) -> "EvaluationCriteria":
        if not self.reward_basis:
            raise ValueError("reward_basis cannot be empty")
        if len(self.reward_basis) != len(set(self.reward_basis)):
            raise ValueError("reward_basis cannot contain duplicates")
        return self


class ScriptedUserTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)
    llm_script: dict[str, tuple[dict[str, Any] | str, ...]] = Field(default_factory=dict)


class UserScenario(BaseModel):
    model_config = ConfigDict(extra="allow")

    known_info: dict[str, Any] | str = Field(default_factory=dict)
    unknown_info: Any = None
    behavior: tuple[str, ...] = ()
    task_instructions: str | None = None
    stop_conditions: tuple[str, ...] = ()
    scripted_turns: tuple[ScriptedUserTurn, ...] = ()


class InitialState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture: str = "default"
    parameters: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    initialization_actions: tuple[dict[str, Any], ...] = ()


class ReferenceAction(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    requestor: str | None = None
    effect: Literal["read", "write"] | None = None


class EvaluationTask(BaseModel):
    """Versioned, outcome-oriented task contract for Smart CS evaluations."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["smart-cs-eval/v1"] = "smart-cs-eval/v1"
    id: str = Field(min_length=1, max_length=256)
    split: Literal["dev", "test", "regression"] = "dev"
    domain: Literal["telecom", "retail", "composite", "shared"]
    capability: str = Field(min_length=1, max_length=128)
    difficulty: Literal["L1", "L2", "L3", "L4"] = "L1"
    risk: Literal["low", "medium", "high", "critical"] = "low"
    tags: tuple[str, ...] = ()
    user_id: str = Field(min_length=1, max_length=128)
    ticket: str = Field(min_length=1, max_length=8000)
    expected_route: str | None = None
    expected_agents: tuple[str, ...] = ()
    expected_capabilities: tuple[str, ...] | None = None
    user_scenario: UserScenario = Field(default_factory=UserScenario)
    initial_state: InitialState = Field(default_factory=InitialState)
    evaluation_criteria: EvaluationCriteria
    reference_actions: tuple[ReferenceAction, ...] = ()
    llm_script: dict[str, tuple[dict[str, Any] | str, ...]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def populate_expected_contract(self) -> "EvaluationTask":
        if self.expected_route is None:
            self.expected_route = self.domain if self.domain != "shared" else None
        if self.expected_capabilities is None:
            self.expected_capabilities = (self.capability,)
        return self


def load_tasks(path: Path) -> list[EvaluationTask]:
    """Load a JSON or YAML task file and validate every task before execution."""

    suffix = path.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ValueError(f"unsupported evaluation task format: {path}")
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]
    if not isinstance(data, list):
        raise ValueError(f"evaluation task file must contain a list: {path}")
    tasks = [EvaluationTask.model_validate(item) for item in data]
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"evaluation task ids must be unique: {path}")
    return tasks
