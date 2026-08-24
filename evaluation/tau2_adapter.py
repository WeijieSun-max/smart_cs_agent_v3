from __future__ import annotations

from typing import Any

from evaluation.harness import EvaluationCase


def from_tau2_task(task: dict[str,Any]) -> EvaluationCase:
    """Map a tau2-style task without coupling production code to its package."""
    return EvaluationCase(
        case_id=str(task.get("id") or task.get("task_id")),
        user_id=str(task.get("user_id") or "eval-user"),
        message=str(task.get("instruction") or task.get("message") or ""),
        expected_tool_sequence=tuple(item.get("name") for item in task.get("expected_actions",[]) if item.get("name")),
        expected_final_state=dict(task.get("expected_state") or {}),
        requires_confirmation=any(item.get("effect")=="write" for item in task.get("expected_actions",[])),
    )
