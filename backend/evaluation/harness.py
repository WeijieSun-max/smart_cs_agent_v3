from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from evaluation.reporting import summarize_results, write_json_report
from evaluation.manifest import build_run_manifest
from evaluation.schema import EvaluationTask, load_tasks
from evaluation.verifiers import verify_task_outcome


class EvaluationCase(BaseModel):
    """A deterministic, machine-checkable contract for one customer-service turn.

    效果断言（trajectory + final state）由 ``EvaluationHarness.run`` 计算；
    延迟预算通过 ``max_llm_calls`` / ``max_node_tokens`` 表达。
    """

    case_id: str
    user_id: str
    message: str
    expected_route: str | None = None
    expected_agents: tuple[str, ...] = ()
    expected_capabilities: tuple[str, ...] = ()
    expected_tool_sequence: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_final_state: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False
    requires_grounding: bool = False
    # 延迟预算：结构化的 LLM 调用次数上限（比墙钟时间更稳定、可回放）。
    max_llm_calls: int | None = None
    max_node_tokens: int | None = None
    # Per-run-name LLM oracle for deterministic replay of manager and child agents.
    llm_script: dict[str, tuple[dict[str, Any] | str, ...]] = Field(default_factory=dict)
    # A case-local replacement for selected in-memory fixture tables.
    initial_state_data: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class RunOutcome(BaseModel):
    """What a runner produces after executing one case (trajectory + state + latency)."""

    run_id: str | None = None
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    environment_before: dict[str, Any] = Field(default_factory=dict)
    environment_after: dict[str, Any] = Field(default_factory=dict)
    turns: list[dict[str, Any]] = Field(default_factory=list)
    llm_calls: int = 0
    llm_call_runs: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    node_latency_ms: dict[str, float] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    case_id: str
    passed: bool
    assertions: dict[str, bool]
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    environment_before: dict[str, Any] = Field(default_factory=dict)
    environment_after: dict[str, Any] = Field(default_factory=dict)
    turns: list[dict[str, Any]] = Field(default_factory=list)
    judge_score: float | None = None
    reward: float = 1.0
    component_scores: dict[str, float] = Field(default_factory=dict)
    vetoes_triggered: tuple[str, ...] = ()
    run_id: str | None = None
    llm_calls: int = 0
    elapsed_ms: float = 0.0


class Judge(Protocol):
    def score(
        self,
        case: EvaluationCase,
        trajectory: list[dict[str, Any]],
        final_state: dict[str, Any],
    ) -> float: ...


Runner = Callable[[EvaluationCase], RunOutcome]
TaskRunner = Callable[[EvaluationTask], RunOutcome]


class EvaluationHarness:
    """Stable replay interface: deterministic assertions over task trajectory and
    final state, plus a deterministic latency budget (llm_call_count).

    Model-judge scoring supplements but never replaces deterministic assertions.
    """

    def __init__(
        self,
        runner: Runner,
        judge: Judge | None = None,
        *,
        task_runner: TaskRunner | None = None,
    ):
        self.runner = runner
        self.judge = judge
        self.task_runner = task_runner

    def run(self, case: EvaluationCase) -> EvaluationResult:
        outcome = self.runner(case)
        trajectory = outcome.trajectory
        final_state = outcome.final_state
        tools = tuple(item.get("tool_name") for item in trajectory if item.get("tool_name"))
        route = final_state.get("intent")
        assignments = final_state.get("agent_assignment_history") or ()
        capabilities = tuple(item.get("capability") for item in assignments if item.get("capability"))
        if not capabilities and route == "fallback":
            capabilities = ("fallback",)
        agents = tuple(dict.fromkeys(item.get("agent") for item in assignments if item.get("agent")))
        grounded = any(
            bool(((item.get("facts") or {}).get("rag") or {}).get("grounded"))
            for item in (final_state.get("task_results") or {}).values()
            if isinstance(item, dict)
        )
        assertions = {
            "route": case.expected_route is None or route == case.expected_route,
            "agents": not case.expected_agents or agents == case.expected_agents,
            "capabilities": not case.expected_capabilities or capabilities == case.expected_capabilities,
            "tool_sequence": not case.expected_tool_sequence or tools == case.expected_tool_sequence,
            "forbidden_tools": not set(tools).intersection(case.forbidden_tools),
            "confirmation": not case.requires_confirmation or final_state.get("pending_action") is not None,
            "grounding": not case.requires_grounding or grounded,
            "final_state": all(final_state.get(key) == value for key, value in case.expected_final_state.items()),
            "llm_calls": case.max_llm_calls is None or outcome.llm_calls <= case.max_llm_calls,
        }
        score = self.judge.score(case, trajectory, final_state) if self.judge else None
        return EvaluationResult(
            case_id=case.case_id,
            passed=all(assertions.values()) and (score is None or score >= 0.7),
            assertions=assertions,
            trajectory=trajectory,
            final_state=final_state,
            environment_before=outcome.environment_before,
            environment_after=outcome.environment_after,
            turns=outcome.turns,
            judge_score=score,
            run_id=outcome.run_id,
            llm_calls=outcome.llm_calls,
            elapsed_ms=outcome.elapsed_ms,
        )

    def run_task(self, task: EvaluationTask) -> EvaluationResult:
        case = task_to_case(task)
        outcome = self.task_runner(task) if self.task_runner is not None else self.runner(case)
        score = self.judge.score(case, outcome.trajectory, outcome.final_state) if self.judge else None
        report = verify_task_outcome(task, outcome, judge_score=score)
        return EvaluationResult(
            case_id=task.id,
            passed=report.passed,
            assertions=report.assertions,
            trajectory=outcome.trajectory,
            final_state=outcome.final_state,
            environment_before=outcome.environment_before,
            environment_after=outcome.environment_after,
            turns=outcome.turns,
            judge_score=score,
            reward=report.reward,
            component_scores=report.component_scores,
            vetoes_triggered=report.vetoes_triggered,
            run_id=outcome.run_id,
            llm_calls=outcome.llm_calls,
            elapsed_ms=outcome.elapsed_ms,
        )


def run_cases(
    cases: list[EvaluationCase],
    runner: Runner,
    judge: Judge | None = None,
) -> list[EvaluationResult]:
    harness = EvaluationHarness(runner, judge)
    return [harness.run(case) for case in cases]


def run_tasks(
    tasks: list[EvaluationTask],
    runner: Runner,
    judge: Judge | None = None,
    *,
    task_runner: TaskRunner | None = None,
) -> list[EvaluationResult]:
    harness = EvaluationHarness(runner, judge, task_runner=task_runner)
    return [harness.run_task(task) for task in tasks]


def task_to_case(task: EvaluationTask) -> EvaluationCase:
    budgets = task.evaluation_criteria.budgets
    return EvaluationCase(
        case_id=task.id,
        user_id=task.user_id,
        message=task.ticket,
        expected_route=task.expected_route,
        expected_agents=task.expected_agents,
        expected_capabilities=task.expected_capabilities or (),
        forbidden_tools=task.evaluation_criteria.forbidden.tools,
        requires_grounding=False,
        max_llm_calls=budgets.max_llm_calls,
        llm_script=task.llm_script,
        initial_state_data=task.initial_state.data,
    )


def _load_cases(path: Path) -> list[EvaluationCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"fixtures must be a JSON list: {path}")
    return [EvaluationCase.model_validate(item) for item in data]


def _load_inputs(path: Path) -> tuple[str, list[EvaluationCase] | list[EvaluationTask]]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)
    items = data.get("tasks") if isinstance(data, dict) and "tasks" in data else data
    if not isinstance(items, list):
        raise ValueError(f"evaluation input must contain a list: {path}")
    if items and all(item.get("schema_version") == "smart-cs-eval/v1" for item in items):
        return "task", load_tasks(path)
    if any(isinstance(item, dict) and item.get("schema_version") for item in items):
        raise ValueError("evaluation files cannot mix schema versions")
    return "case", [EvaluationCase.model_validate(item) for item in items]


def _summarize(results: list[EvaluationResult]) -> str:
    lines: list[str] = []
    lines.append(f"{'case_id':36} {'passed':7} {'llm_calls':9} {'elapsed_ms':10}  failed-assertions")
    lines.append("-" * 100)
    for result in results:
        failed = ", ".join(name for name, ok in result.assertions.items() if not ok) or "-"
        lines.append(
            f"{result.case_id:36} {str(result.passed):7} {result.llm_calls:9} {result.elapsed_ms:10.1f}  {failed}"
        )
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    calls = [result.llm_calls for result in results]
    lat = [result.elapsed_ms for result in results]
    lines.append("-" * 100)
    lines.append(f"passed {passed}/{total}")
    if calls:
        calls_sorted = sorted(calls)
        lines.append(
            "llm_calls  min/median/max = "
            f"{calls_sorted[0]} / {calls_sorted[len(calls_sorted) // 2]} / {calls_sorted[-1]}"
        )
    if lat:
        lat_sorted = sorted(lat)
        lines.append(
            "elapsed_ms min/median/max = "
            f"{lat_sorted[0]:.1f} / {lat_sorted[len(lat_sorted) // 2]:.1f} / {lat_sorted[-1]:.1f}"
        )
    structured = summarize_results(results)
    if structured["component_pass_rates"]:
        components = ", ".join(
            f"{name}={value:.3f}"
            for name, value in structured["component_pass_rates"].items()
        )
        lines.append(f"components  {components}")
    if structured["veto_counts"]:
        lines.append(f"vetoes      {structured['veto_counts']}")
    reliability = structured["reliability"]
    if reliability["pass_at_k"] is not None:
        lines.append(
            f"reliability k={reliability['k']} "
            f"Pass@k={reliability['pass_at_k']:.3f} "
            f"Pass^k={reliability['pass_power_k']:.3f}"
        )
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: python -m evaluation.harness <cases.json|yaml> "
            "[--trials K] [--json-output report.json]",
            file=sys.stderr,
        )
        return 2
    fixtures_path = Path(argv[1])
    kind, inputs = _load_inputs(fixtures_path)
    output_path: Path | None = None
    if "--json-output" in argv:
        index = argv.index("--json-output")
        if index + 1 >= len(argv):
            print("--json-output requires a path", file=sys.stderr)
            return 2
        output_path = Path(argv[index + 1])
    trials = 1
    if "--trials" in argv:
        index = argv.index("--trials")
        if index + 1 >= len(argv):
            print("--trials requires a positive integer", file=sys.stderr)
            return 2
        try:
            trials = int(argv[index + 1])
        except ValueError:
            trials = 0
        if trials < 1:
            print("--trials requires a positive integer", file=sys.stderr)
            return 2

    # Import the deterministic in-memory runner lazily so harness.py stays lightweight.
    from evaluation.deterministic_runner import run_case, run_task_conversation

    started = time.perf_counter()
    if kind == "task":
        results = []
        for _ in range(trials):
            results.extend(run_tasks(  # type: ignore[arg-type]
                inputs,
                run_case,
                task_runner=run_task_conversation,
            ))
    else:
        results = []
        for _ in range(trials):
            results.extend(run_cases(inputs, run_case))  # type: ignore[arg-type]
    print(_summarize(results))
    print(f"total_elapsed_ms={((time.perf_counter() - started) * 1000):.1f}")
    if output_path is not None:
        manifest = build_run_manifest(
            fixtures_path,
            trials=trials,
            runner="deterministic_conversation" if kind == "task" else "deterministic_case",
            environment="in_memory",
            repository_root=Path(__file__).resolve().parents[1],
        )
        write_json_report(output_path, results, manifest=manifest)
        print(f"json_report={output_path}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
