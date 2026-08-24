from __future__ import annotations

import math
from collections import defaultdict
from threading import Lock


_SKILLS = frozenset(
    {
        "onboarding_process_guide",
        "onboarding_material_check",
        "onboarding_risk_assessment",
        "onboarding_eligibility_check",
        "none",
    }
)
_RUNTIME_OUTCOMES = frozenset(
    {
        "completed",
        "needs_clarification",
        "awaiting_confirmation",
        "indeterminate",
        "budget_exhausted",
        "failed",
    }
)
_PENDING_OUTCOMES = frozenset(
    {"created", "confirmed", "rejected", "expired", "duplicate", "failed"}
)
_SECURITY_OUTCOMES = frozenset(
    {"allowlist_denied", "schema_invalid", "unauthorized_write"}
)
_NUMERIC_MAX = 9_007_199_254_740_991


def _bounded_int(value: object, maximum: int) -> int | None:
    if type(value) is not int or not 0 <= value <= maximum:
        return None
    return value


def _bounded_number(value: object, maximum: float) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(number) or not 0 <= number <= maximum:
        return None
    return number


def _saturating_add(
    values: dict[str, int | float], key: str, amount: int | float
) -> None:
    current = values[key]
    total = current + amount
    if not math.isfinite(float(total)) or total > _NUMERIC_MAX:
        total = _NUMERIC_MAX
    values[key] = total


class SkillMetrics:
    """Small numeric-only registry whose labels can never come from user/model text."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._selection: dict[str, int | float] = defaultdict(int)
        self._runtime: dict[str, int | float] = defaultdict(int)
        self._pending: dict[str, int] = defaultdict(int)
        self._security: dict[str, int] = defaultdict(int)

    def record_selection(self, skill: str, *, selected: bool, confidence: float) -> None:
        confidence_value = _bounded_number(confidence, 1.0)
        if (
            type(skill) is not str
            or skill not in _SKILLS
            or type(selected) is not bool
            or confidence_value is None
            or confidence_value > 1
        ):
            return
        outcome = "selected" if selected else "not_selected"
        with self._lock:
            _saturating_add(self._selection, f"{skill}.{outcome}", 1)
            _saturating_add(
                self._selection, f"{skill}.confidence_total", confidence_value
            )

    def record_runtime(
        self,
        outcome: str,
        *,
        llm_decisions: int,
        tool_calls: int,
        elapsed_ms: int | float,
    ) -> None:
        llm_value = _bounded_int(llm_decisions, 3)
        tool_value = _bounded_int(tool_calls, 5)
        elapsed_value = _bounded_number(elapsed_ms, 30_000)
        if (
            type(outcome) is not str
            or outcome not in _RUNTIME_OUTCOMES
            or llm_value is None
            or tool_value is None
            or elapsed_value is None
        ):
            return
        with self._lock:
            _saturating_add(self._runtime, f"{outcome}.count", 1)
            _saturating_add(
                self._runtime, f"{outcome}.llm_decisions_total", llm_value
            )
            _saturating_add(
                self._runtime, f"{outcome}.tool_calls_total", tool_value
            )
            _saturating_add(
                self._runtime, f"{outcome}.elapsed_ms_total", elapsed_value
            )
            maximum = f"{outcome}.elapsed_ms_max"
            self._runtime[maximum] = max(self._runtime[maximum], elapsed_value)

    def record_pending(self, outcome: str, count: int = 1) -> None:
        count_value = _bounded_int(count, 10_000)
        if (
            type(outcome) is not str
            or outcome not in _PENDING_OUTCOMES
            or count_value is None
        ):
            return
        with self._lock:
            _saturating_add(self._pending, outcome, count_value)

    def record_security(self, outcome: str) -> None:
        if type(outcome) is not str or outcome not in _SECURITY_OUTCOMES:
            return
        with self._lock:
            _saturating_add(self._security, outcome, 1)

    def snapshot(self) -> dict[str, dict[str, int | float]]:
        with self._lock:
            return {
                "selection": dict(sorted(self._selection.items())),
                "runtime": dict(sorted(self._runtime.items())),
                "pending": dict(sorted(self._pending.items())),
                "security": dict(sorted(self._security.items())),
            }

    def reset(self) -> None:
        with self._lock:
            self._selection.clear()
            self._runtime.clear()
            self._pending.clear()
            self._security.clear()


skill_metrics = SkillMetrics()


def skill_metrics_snapshot() -> dict[str, dict[str, int | float]]:
    return skill_metrics.snapshot()


__all__ = ["SkillMetrics", "skill_metrics", "skill_metrics_snapshot"]
