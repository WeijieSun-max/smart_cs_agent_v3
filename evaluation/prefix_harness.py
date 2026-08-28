from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from evaluation.manifest import build_run_manifest
from evaluation.prefix import (
    PrefixEvaluationResult,
    TrajectoryPrefixTask,
    load_prefix_tasks,
    run_supervisor_prefix_async,
    verify_prefix_decision,
)
from evaluation.reporting import sanitize_report_value


def load_recorded_decisions(path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = raw.get("decisions") if isinstance(raw, dict) and "decisions" in raw else raw
    if isinstance(data, dict):
        decisions = data
    elif isinstance(data, list):
        decisions = {
            str(item["task_id"]): item["decision"]
            for item in data
            if isinstance(item, dict) and "task_id" in item and "decision" in item
        }
        if len(decisions) != len(data):
            raise ValueError("each recorded decision item requires task_id and decision")
    else:
        raise ValueError(f"recorded decisions must be a mapping or list: {path}")
    if not all(isinstance(value, dict) for value in decisions.values()):
        raise ValueError("each recorded decision must be a mapping")
    return {str(key): value for key, value in decisions.items()}


def run_recorded_prefix_tasks(
    tasks: list[TrajectoryPrefixTask],
    decisions: dict[str, dict[str, Any]],
) -> list[PrefixEvaluationResult]:
    task_ids = {task.id for task in tasks}
    missing = task_ids - set(decisions)
    extra = set(decisions) - task_ids
    if missing or extra:
        raise ValueError(
            f"recorded decision ids do not match tasks; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return [verify_prefix_decision(task, decisions[task.id]) for task in tasks]


async def run_live_supervisor_tasks(
    tasks: list[TrajectoryPrefixTask],
) -> list[PrefixEvaluationResult]:
    unsupported = [task.id for task in tasks if task.decision_level != "supervisor"]
    if unsupported:
        raise ValueError(
            "live supervisor runner received domain-agent tasks: " + ", ".join(unsupported)
        )
    return [await run_supervisor_prefix_async(task) for task in tasks]


def write_prefix_report(
    path: Path,
    results: list[PrefixEvaluationResult],
    *,
    manifest: Any,
) -> None:
    passed = sum(result.passed for result in results)
    payload = {
        "manifest": sanitize_report_value(manifest.model_dump(mode="python")),
        "summary": {
            "task_count": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": passed / len(results) if results else 1.0,
        },
        "results": [
            sanitize_report_value(result.model_dump(mode="python"))
            for result in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen trajectory-prefix decisions")
    parser.add_argument("tasks", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--decisions", type=Path, help="JSON/YAML decisions keyed by task id")
    source.add_argument(
        "--live-supervisor",
        action="store_true",
        help="initialize configured model profiles and call the real supervisor boundary",
    )
    parser.add_argument(
        "--decision-level",
        choices=("supervisor", "domain_agent"),
        help="filter tasks before running",
    )
    parser.add_argument("--json-output", type=Path)
    return parser


def _main(argv: list[str]) -> int:
    args = _parser().parse_args(argv[1:])
    tasks = load_prefix_tasks(args.tasks)
    if args.decision_level:
        tasks = [task for task in tasks if task.decision_level == args.decision_level]
    if not tasks:
        print("no prefix tasks selected", file=sys.stderr)
        return 2

    try:
        if args.live_supervisor:
            from infra.llm.bootstrap import initialize_llm

            initialize_llm()
            results = asyncio.run(run_live_supervisor_tasks(tasks))
            runner_name = "live_supervisor_prefix"
            model_profiles = _configured_model_profiles()
            prompt_versions = {"supervisor.decide": "manager-v2-structured-context"}
        else:
            decisions = load_recorded_decisions(args.decisions)
            if args.decision_level:
                selected_ids = {task.id for task in tasks}
                decisions = {
                    task_id: decision
                    for task_id, decision in decisions.items()
                    if task_id in selected_ids
                }
            results = run_recorded_prefix_tasks(tasks, decisions)
            runner_name = "recorded_prefix_decisions"
            model_profiles = {}
            prompt_versions = {}
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"prefix evaluation failed: {exc}", file=sys.stderr)
        return 2

    for result in results:
        failed = ",".join(name for name, ok in result.assertions.items() if not ok) or "-"
        print(f"{result.task_id:48} passed={str(result.passed):5} failed={failed}")
    passed = sum(result.passed for result in results)
    print(f"passed {passed}/{len(results)}")

    if args.json_output:
        manifest = build_run_manifest(
            args.tasks,
            trials=1,
            runner=runner_name,
            environment="frozen_prefix",
            repository_root=Path(__file__).resolve().parents[1],
            model_profiles=model_profiles,
            prompt_versions=prompt_versions,
        )
        write_prefix_report(args.json_output, results, manifest=manifest)
        print(f"json_report={args.json_output}")
    return 0 if passed == len(results) else 1


def _configured_model_profiles() -> dict[str, str]:
    from pkg.config.settings import get_settings

    settings = get_settings()
    default = settings.qwen_model
    return {
        "default": default,
        "supervisor": settings.qwen_supervisor_model or default,
    }


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
