from __future__ import annotations

from collections.abc import Callable
from typing import Any,Protocol

from pydantic import BaseModel,Field


class EvaluationCase(BaseModel):
    case_id: str
    user_id: str
    message: str
    expected_route: str|None=None
    expected_tool_sequence: tuple[str,...]=()
    forbidden_tools: tuple[str,...]=()
    expected_final_state: dict[str,Any]=Field(default_factory=dict)
    requires_confirmation: bool=False


class EvaluationResult(BaseModel):
    case_id: str
    passed: bool
    assertions: dict[str,bool]
    trajectory: list[dict[str,Any]]=Field(default_factory=list)
    final_state: dict[str,Any]=Field(default_factory=dict)
    judge_score: float|None=None


class Judge(Protocol):
    def score(self,case: EvaluationCase,trajectory: list[dict[str,Any]],final_state: dict[str,Any]) -> float: ...


class EvaluationHarness:
    """Stable replay interface for deterministic, tau2 and optional judge adapters."""
    def __init__(self,runner: Callable[[EvaluationCase],tuple[list[dict[str,Any]],dict[str,Any]]],judge: Judge|None=None):
        self.runner=runner; self.judge=judge

    def run(self,case: EvaluationCase) -> EvaluationResult:
        trajectory,final_state=self.runner(case); tools=tuple(item.get("tool_name") for item in trajectory if item.get("tool_name")); route=final_state.get("intent")
        assertions={
            "route": case.expected_route is None or route==case.expected_route,
            "tool_sequence": not case.expected_tool_sequence or tools==case.expected_tool_sequence,
            "forbidden_tools": not set(tools).intersection(case.forbidden_tools),
            "confirmation": not case.requires_confirmation or final_state.get("pending_action") is not None,
            "final_state": all(final_state.get(key)==value for key,value in case.expected_final_state.items()),
        }
        score=self.judge.score(case,trajectory,final_state) if self.judge else None
        return EvaluationResult(case_id=case.case_id,passed=all(assertions.values()) and (score is None or score>=0.7),assertions=assertions,trajectory=trajectory,final_state=final_state,judge_score=score)
