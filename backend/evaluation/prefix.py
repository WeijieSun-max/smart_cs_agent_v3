from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


DecisionLevel = Literal["supervisor", "domain_agent"]
DecisionAction = Literal[
    "dispatch",
    "finish",
    "clarify",
    "confirm_action",
    "reject_action",
    "tool_call",
    "tool_calls",
    "propose_write",
    "final",
]


class PrefixActionSpec(BaseModel):
    """One acceptable or prohibited next action after a frozen trajectory prefix."""

    model_config = ConfigDict(extra="forbid")

    action: DecisionAction
    agent: str | None = Field(default=None, max_length=128)
    capability: str | None = Field(default=None, max_length=128)
    tool_name: str | None = Field(default=None, max_length=128)
    arguments_contains: dict[str, Any] = Field(default_factory=dict)


class TrajectoryPrefixTask(BaseModel):
    """A tau-bench-style next-action task with multiple valid continuations."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["smart-cs-prefix-eval/v1"] = "smart-cs-prefix-eval/v1"
    id: str = Field(min_length=1, max_length=256)
    split: Literal["dev", "test", "regression"] = "dev"
    decision_level: DecisionLevel = "supervisor"
    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=8000)
    frozen_state: dict[str, Any] = Field(default_factory=dict)
    active_action: dict[str, Any] | None = None
    allow_dispatch: bool = True
    allowed_actions: tuple[PrefixActionSpec, ...] = Field(min_length=1)
    forbidden_actions: tuple[PrefixActionSpec, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_identical_allowed_and_forbidden_actions(self) -> "TrajectoryPrefixTask":
        allowed = {item.model_dump_json() for item in self.allowed_actions}
        forbidden = {item.model_dump_json() for item in self.forbidden_actions}
        if allowed.intersection(forbidden):
            raise ValueError("the same action cannot be both allowed and forbidden")
        return self


class PrefixEvaluationResult(BaseModel):
    task_id: str
    passed: bool
    assertions: dict[str, bool]
    decision: dict[str, Any]
    matched_allowed_indices: tuple[int, ...] = ()
    matched_forbidden_indices: tuple[int, ...] = ()


PrefixDecisionRunner = Callable[
    [TrajectoryPrefixTask],
    dict[str, Any] | BaseModel | Awaitable[dict[str, Any] | BaseModel],
]


def verify_prefix_decision(
    task: TrajectoryPrefixTask,
    decision: dict[str, Any] | BaseModel,
) -> PrefixEvaluationResult:
    """Pass when any allowed action matches and no forbidden action matches."""

    normalized = _decision_mapping(decision)
    allowed = tuple(
        index
        for index, action in enumerate(task.allowed_actions)
        if _action_matches(action, normalized)
    )
    forbidden = tuple(
        index
        for index, action in enumerate(task.forbidden_actions)
        if _action_matches(action, normalized)
    )
    assertions = {
        "matches_allowed_action": bool(allowed),
        "avoids_forbidden_action": not forbidden,
    }
    return PrefixEvaluationResult(
        task_id=task.id,
        passed=all(assertions.values()),
        assertions=assertions,
        decision=normalized,
        matched_allowed_indices=allowed,
        matched_forbidden_indices=forbidden,
    )


async def run_prefix_task_async(
    task: TrajectoryPrefixTask,
    runner: PrefixDecisionRunner,
) -> PrefixEvaluationResult:
    output = runner(task)
    if inspect.isawaitable(output):
        output = await output
    return verify_prefix_decision(task, output)


async def run_supervisor_prefix_async(
    task: TrajectoryPrefixTask,
    *,
    decider: Callable[..., Awaitable[BaseModel]] | None = None,
) -> PrefixEvaluationResult:
    """Exercise the real supervisor decision boundary using the configured LLM.

    Model initialization is deliberately external: deterministic tests can install a
    scripted model and live evaluations can install the normal profile registry.
    """

    if task.decision_level != "supervisor":
        raise ValueError("supervisor prefix runner requires decision_level=supervisor")
    from domain.customer_service_agent.agents.supervisor_agent import decide_next_step
    from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state

    state = create_chat_state(
        task.user_id,
        f"prefix-eval-{task.id}",
        task.query,
        turn_id=f"prefix-eval-{task.id}",
    )
    state.update(deepcopy(task.frozen_state))
    decision_fn = decider or decide_next_step
    decision = await decision_fn(
        state,
        active_action=deepcopy(task.active_action),
        allow_dispatch=task.allow_dispatch,
    )
    return verify_prefix_decision(task, decision)


def load_prefix_tasks(path: Path) -> list[TrajectoryPrefixTask]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = raw.get("tasks") if isinstance(raw, dict) and "tasks" in raw else raw
    if not isinstance(data, list):
        raise ValueError(f"prefix evaluation file must contain a list: {path}")
    tasks = [TrajectoryPrefixTask.model_validate(item) for item in data]
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"prefix evaluation task ids must be unique: {path}")
    return tasks


def _decision_mapping(decision: dict[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(decision, BaseModel):
        return decision.model_dump(mode="python")
    if not isinstance(decision, dict):
        raise TypeError("prefix decision must be a mapping or Pydantic model")
    return deepcopy(decision)


def _action_matches(spec: PrefixActionSpec, decision: dict[str, Any]) -> bool:
    if decision.get("action") != spec.action:
        return False
    candidates = _candidate_payloads(decision)
    return any(_payload_matches(spec, candidate) for candidate in candidates)


def _candidate_payloads(decision: dict[str, Any]) -> list[dict[str, Any]]:
    action = decision.get("action")
    if action == "dispatch":
        assignments = decision.get("assignments") or []
        return [item for item in assignments if isinstance(item, dict)] or [{}]
    if action == "tool_calls":
        calls = decision.get("tool_calls") or []
        return [item for item in calls if isinstance(item, dict)] or [{}]
    return [decision]


def _payload_matches(spec: PrefixActionSpec, payload: dict[str, Any]) -> bool:
    for key in ("agent", "capability", "tool_name"):
        expected = getattr(spec, key)
        if expected is not None and payload.get(key) != expected:
            return False
    arguments = payload.get("arguments") or {}
    return _contains(arguments, spec.arguments_contains)


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, (list, tuple)) and all(
            any(_contains(candidate, value) for candidate in actual)
            for value in expected
        )
    return actual == expected
