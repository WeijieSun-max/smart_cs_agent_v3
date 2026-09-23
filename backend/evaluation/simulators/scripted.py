from __future__ import annotations

from typing import Any

from evaluation.schema import EvaluationTask, ScriptedUserTurn
from evaluation.simulators.base import UserTurnDecision


class ScriptedUserSimulator:
    """Deterministic simulator backed by ``user_scenario.scripted_turns``."""

    def __init__(self) -> None:
        self._turns: tuple[ScriptedUserTurn, ...] = ()
        self._cursor = 0

    def reset(self, task: EvaluationTask, initial_observation: dict[str, Any]) -> None:
        del initial_observation
        self._turns = task.user_scenario.scripted_turns
        self._cursor = 0

    def next_turn(
        self,
        assistant_message: str,
        observation: dict[str, Any],
    ) -> UserTurnDecision:
        del assistant_message, observation
        if self._cursor >= len(self._turns):
            return UserTurnDecision(stop=True, reason="script_exhausted")
        turn = self._turns[self._cursor]
        self._cursor += 1
        return UserTurnDecision(
            message=turn.message,
            stop=False,
            reason="scripted_turn",
            llm_script=turn.llm_script,
        )
