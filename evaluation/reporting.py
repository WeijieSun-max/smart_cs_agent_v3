from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pkg.telemetry.langfuse_observability import redact_sensitive_text


_PII_KEYS = {
    "phone",
    "email",
    "recipient",
    "detail",
    "full_address",
}


def summarize_results(results: list[Any]) -> dict[str, Any]:
    total = len(results)
    passed = sum(bool(item.passed) for item in results)
    component_values: dict[str, list[float]] = {}
    latencies: list[float] = []
    llm_calls: list[int] = []
    veto_counts: dict[str, int] = {}

    for result in results:
        latencies.append(float(result.elapsed_ms))
        llm_calls.append(int(result.llm_calls))
        for name, value in (result.component_scores or {}).items():
            component_values.setdefault(name, []).append(float(value))
        for veto in result.vetoes_triggered or ():
            veto_counts[veto] = veto_counts.get(veto, 0) + 1

    return {
        "case_count": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 1.0,
        "component_pass_rates": {
            name: sum(values) / len(values)
            for name, values in sorted(component_values.items())
        },
        "llm_calls": {
            "min": min(llm_calls, default=0),
            "median": _percentile(llm_calls, 0.5),
            "max": max(llm_calls, default=0),
        },
        "elapsed_ms": {
            "min": min(latencies, default=0.0),
            "median": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=0.0),
        },
        "veto_counts": dict(sorted(veto_counts.items())),
        "reliability": reliability_metrics(results),
    }


def reliability_metrics(results: list[Any]) -> dict[str, Any]:
    """Compute Pass@k and Pass^k from repeated runs sharing a case id."""

    grouped: dict[str, list[bool]] = {}
    for result in results:
        grouped.setdefault(str(result.case_id), []).append(bool(result.passed))
    repeated = {case_id: values for case_id, values in grouped.items() if len(values) > 1}
    if not repeated:
        return {"k": 1, "task_count": len(grouped), "pass_at_k": None, "pass_power_k": None}
    sizes = {len(values) for values in repeated.values()}
    k: int | None = next(iter(sizes)) if len(sizes) == 1 else None
    return {
        "k": k,
        "task_count": len(repeated),
        "pass_at_k": sum(any(values) for values in repeated.values()) / len(repeated),
        "pass_power_k": sum(all(values) for values in repeated.values()) / len(repeated),
        "per_task": {
            case_id: {
                "trials": len(values),
                "pass_count": sum(values),
                "pass_at_k": any(values),
                "pass_power_k": all(values),
            }
            for case_id, values in sorted(repeated.items())
        },
    }


def write_json_report(path: Path, results: list[Any]) -> None:
    payload = {
        "summary": summarize_results(results),
        "results": [_sanitize_report_value(item.model_dump(mode="python")) for item in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _percentile(values: list[float] | list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return round(ordered[index], 3)


def _sanitize_report_value(value: Any, *, key: str = "") -> Any:
    normalized = key.strip().lower()
    if normalized in _PII_KEYS or normalized.endswith(("_cipher", "_hash")):
        return "[REDACTED_PII]"
    if isinstance(value, dict):
        return {
            str(item_key): _sanitize_report_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_report_value(item) for item in value]
    if isinstance(value, bytes):
        return "[REDACTED_BINARY]"
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return value
