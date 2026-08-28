from __future__ import annotations

import inspect
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from evaluation.schema import EvaluationTask
from evaluation.simulators.base import UserTurnDecision


class UngroundedUserSimulationError(ValueError):
    pass


class UserSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_instructions: str | None = None
    behavior: tuple[str, ...] = ()
    stop_conditions: tuple[str, ...] = ()
    known_facts: dict[str, Any] = Field(default_factory=dict)
    authorized_observations: dict[str, Any] = Field(default_factory=dict)
    transcript: tuple[dict[str, str], ...] = ()
    latest_assistant_message: str


class UserSimulationModel(Protocol):
    def __call__(self, request: UserSimulationRequest) -> UserTurnDecision | dict[str, Any]: ...


class GroundedLLMUserSimulator:
    """LLM user policy constrained to an explicit fact ledger.

    The model must return every concrete fact used in ``grounded_facts``. The
    simulator rejects facts absent from the task's known information or the
    environment observations explicitly exposed to the user simulator.
    """

    def __init__(
        self,
        model: UserSimulationModel,
        *,
        observation_filter: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._model = model
        self._observation_filter = observation_filter or (lambda _: {})
        self._task: EvaluationTask | None = None
        self._known_facts: dict[str, Any] = {}
        self._transcript: list[dict[str, str]] = []

    def reset(self, task: EvaluationTask, initial_observation: dict[str, Any]) -> None:
        self._task = task
        known = task.user_scenario.known_info
        self._known_facts = deepcopy(known if isinstance(known, dict) else {"known_info": known})
        self._transcript = []
        # Validate the filter at reset time, without implicitly exposing its output.
        exposed = self._observation_filter(deepcopy(initial_observation))
        if not isinstance(exposed, dict):
            raise TypeError("user simulator observation_filter must return a mapping")

    def next_turn(
        self,
        assistant_message: str,
        observation: dict[str, Any],
    ) -> UserTurnDecision:
        if self._task is None:
            raise RuntimeError("user simulator must be reset before use")
        exposed = self._observation_filter(deepcopy(observation))
        if not isinstance(exposed, dict):
            raise TypeError("user simulator observation_filter must return a mapping")
        request = UserSimulationRequest(
            task_id=self._task.id,
            task_instructions=self._task.user_scenario.task_instructions,
            behavior=self._task.user_scenario.behavior,
            stop_conditions=self._task.user_scenario.stop_conditions,
            known_facts=deepcopy(self._known_facts),
            authorized_observations=exposed,
            transcript=tuple(self._transcript),
            latest_assistant_message=assistant_message,
        )
        raw = self._model(request)
        if inspect.isawaitable(raw):
            raise TypeError("GroundedLLMUserSimulator requires a synchronous model adapter")
        decision = raw if isinstance(raw, UserTurnDecision) else UserTurnDecision.model_validate(raw)
        allowed = _flatten({"known": self._known_facts, "observed": exposed})
        claimed = _flatten({"known": decision.grounded_facts.get("known", {}),
                            "observed": decision.grounded_facts.get("observed", {})})
        invalid = {
            path: value
            for path, value in claimed.items()
            if path not in allowed or allowed[path] != value
        }
        if invalid:
            paths = ", ".join(sorted(invalid))
            raise UngroundedUserSimulationError(f"simulated user invented or altered facts: {paths}")
        if not decision.stop and not (decision.message or "").strip():
            raise ValueError("continuing user simulation requires a non-empty message")
        self._transcript.extend([
            {"role": "assistant", "content": assistant_message},
            {"role": "user", "content": decision.message or ""},
        ])
        return decision


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(item, path))
        return output
    return {prefix: value}
