from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from evaluation.harness import RunOutcome
from evaluation.schema import EvaluationTask, InitialState
from evaluation.simulators import UserSimulator


class IsolatedEnvironment(Protocol):
    def assert_isolated(self) -> None: ...
    def reset(self, initial_state: InitialState) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...


class LiveTurnResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_message: str
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    llm_calls: int = Field(default=0, ge=0)
    llm_call_runs: list[str] = Field(default_factory=list)


LiveTurnExecutor = Callable[[str, str, str, str], LiveTurnResult | dict[str, Any]]
SimulatorFactory = Callable[[EvaluationTask], UserSimulator]


class LiveConversationRunner:
    """Injectable runner for real model profiles and an isolated environment.

    ``turn_executor`` may wrap the application chat service or graph. This class
    owns evaluation reset, simulator turns, snapshots, and accounting while model
    and infrastructure bootstrapping remain explicit at the composition root.
    """

    def __init__(
        self,
        *,
        environment: IsolatedEnvironment,
        turn_executor: LiveTurnExecutor,
        simulator_factory: SimulatorFactory,
        model_bootstrap: Callable[[], None] | None = None,
        max_turns: int = 20,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self._environment = environment
        self._turn_executor = turn_executor
        self._simulator_factory = simulator_factory
        self._model_bootstrap = model_bootstrap
        self._max_turns = max_turns
        self._bootstrapped = False

    def __call__(self, task: EvaluationTask) -> RunOutcome:
        self._environment.assert_isolated()
        self._environment.reset(task.initial_state)
        if self._model_bootstrap is not None and not self._bootstrapped:
            self._model_bootstrap()
            self._bootstrapped = True
        before = self._environment.snapshot()
        simulator = self._simulator_factory(task)
        simulator.reset(task, before)
        run_id = uuid4().hex
        session_id = f"live-eval-{task.id}-{run_id[:8]}"
        message = task.ticket
        trajectory: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        llm_runs: list[str] = []
        llm_call_count = 0
        final_state: dict[str, Any] = {}
        started = time.perf_counter()
        task_turn_budget = task.evaluation_criteria.budgets.max_turns
        turn_limit = min(self._max_turns, task_turn_budget or self._max_turns)

        for turn_index in range(turn_limit):
            turn_id = f"live-eval-{run_id[:10]}-{turn_index}"
            raw = self._turn_executor(task.user_id, session_id, turn_id, message)
            result = raw if isinstance(raw, LiveTurnResult) else LiveTurnResult.model_validate(raw)
            trajectory.extend(result.trajectory)
            trajectory.append({
                "phase": "assistant_message",
                "turn_index": turn_index,
                "content": result.assistant_message,
            })
            final_state = result.final_state
            llm_call_count += result.llm_calls
            llm_runs.extend(result.llm_call_runs)
            turns.append({
                "turn_index": turn_index,
                "turn_id": turn_id,
                "user_message": message,
                "assistant_message": result.assistant_message,
            })
            user = simulator.next_turn(result.assistant_message, self._environment.snapshot())
            if user.stop:
                break
            message = user.message or ""
        else:
            trajectory.append({"phase": "budget_exhausted", "budget": "max_turns"})

        after = self._environment.snapshot()
        return RunOutcome(
            run_id=run_id,
            trajectory=trajectory,
            final_state=final_state,
            environment_before=before,
            environment_after=after,
            turns=turns,
            llm_calls=llm_call_count,
            llm_call_runs=llm_runs,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
