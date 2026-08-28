from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from evaluation.schema import EvaluationTask


class UserTurnDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str | None = Field(default=None, max_length=8000)
    stop: bool = False
    reason: str = Field(default="continue", max_length=256)
    grounded_facts: dict[str, Any] = Field(default_factory=dict)
    llm_script: dict[str, tuple[dict[str, Any] | str, ...]] = Field(default_factory=dict)


class UserSimulator(Protocol):
    """Stateful user policy used after the task's initial ticket."""

    def reset(self, task: EvaluationTask, initial_observation: dict[str, Any]) -> None: ...

    def next_turn(
        self,
        assistant_message: str,
        observation: dict[str, Any],
    ) -> UserTurnDecision: ...
