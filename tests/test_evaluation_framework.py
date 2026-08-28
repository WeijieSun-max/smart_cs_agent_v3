from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.deterministic_runner import (
    default_fixtures,
    run_case,
    run_task_conversation,
)
from evaluation.harness import EvaluationCase, EvaluationResult, RunOutcome, run_tasks
from evaluation.environments import MySQLEvaluationEnvironment
from evaluation.manifest import build_run_manifest, file_sha256
from evaluation.prefix import (
    load_prefix_tasks,
    run_supervisor_prefix_async,
    verify_prefix_decision,
)
from evaluation.prefix_harness import load_recorded_decisions, run_recorded_prefix_tasks
from evaluation.reporting import (
    reliability_metrics,
    summarize_results,
    wilson_interval,
    write_json_report,
)
from evaluation.runners import LiveConversationRunner
from evaluation.schema import EvaluationTask, load_tasks
from evaluation.simulators import (
    GroundedLLMUserSimulator,
    ScriptedUserSimulator,
    UngroundedUserSimulationError,
)
from evaluation.tau2_adapter import from_tau2_task, import_tau2_task
from evaluation.verifiers import verify_task_outcome


GOLD_CASES = Path(__file__).parents[1] / "evaluation" / "datasets" / "dev" / "gold_cases.json"
PREFIX_CASES = Path(__file__).parents[1] / "evaluation" / "datasets" / "dev" / "prefix_cases.json"
PREFIX_DECISIONS = (
    Path(__file__).parents[1]
    / "evaluation"
    / "datasets"
    / "dev"
    / "prefix_candidate_decisions.json"
)


def _task(**overrides) -> EvaluationTask:
    payload = {
        "id": "eval-test",
        "domain": "telecom",
        "capability": "usage",
        "user_id": "u-active",
        "ticket": "查询流量",
        "expected_route": None,
        "expected_capabilities": [],
        "evaluation_criteria": {
            "state_assertions": [
                {"path": "lines.L1.current_plan_id", "op": "eq", "expected": "P1"}
            ],
            "reward_basis": ["STATE"],
        },
    }
    payload.update(overrides)
    return EvaluationTask.model_validate(payload)


def test_v1_schema_loads_gold_cases_and_rejects_duplicate_reward_components() -> None:
    tasks = load_tasks(GOLD_CASES)

    assert len(tasks) == 6
    assert len({task.id for task in tasks}) == 6
    assert tasks[0].schema_version == "smart-cs-eval/v1"

    with pytest.raises(ValidationError, match="duplicates"):
        _task(evaluation_criteria={"reward_basis": ["STATE", "STATE"]})


def test_tau2_import_reads_current_schema_without_making_actions_an_exact_path() -> None:
    source = {
        "id": "tau-task-1",
        "ticket": "The user wants help with mobile data.",
        "user_scenario": {
            "instructions": {
                "known_info": "The user is abroad.",
                "unknown_info": "Roaming is off.",
                "task_instructions": "Do not invent device observations.",
            }
        },
        "initial_state": {
            "initialization_actions": [
                {"env_type": "user", "func_name": "turn_roaming_off"}
            ]
        },
        "evaluation_criteria": {
            "actions": [{"name": "enable_roaming", "arguments": {"line_id": "L1"}}],
            "env_assertions": [{"func_name": "assert_mobile_data_status"}],
            "communicate_info": ["roaming"],
            "reward_basis": ["DB", "COMMUNICATE"],
        },
    }

    imported = import_tau2_task(source, domain="telecom", capability="roaming")
    legacy = from_tau2_task(source)

    assert imported.ticket == source["ticket"]
    assert imported.reference_actions[0].name == "enable_roaming"
    assert imported.initial_state.initialization_actions[0]["env_type"] == "user"
    assert imported.metadata["requires_domain_state_mapper"] is True
    assert legacy.message == source["ticket"]
    assert legacy.expected_tool_sequence == ()


def test_reference_actions_are_diagnostic_unless_action_gates_reward() -> None:
    task = _task(reference_actions=[{"name": "a_different_valid_read"}])
    outcome = RunOutcome(
        trajectory=[{"tool_name": "actual_read", "phase": "tool_result"}],
        final_state={"intent": "telecom"},
        environment_after={"lines": {"L1": {"current_plan_id": "P1"}}},
    )

    report = verify_task_outcome(task, outcome)

    assert report.passed is True
    assert report.reward == 1.0
    assert "ACTION" not in report.component_scores
    assert report.diagnostics["reference_action_names"] == ["a_different_valid_read"]


def test_confirmation_bypass_is_a_hard_veto() -> None:
    task = _task(
        evaluation_criteria={
            "vetoes": ["confirmation_bypass"],
            "reward_basis": ["SAFETY"],
        }
    )
    outcome = RunOutcome(
        trajectory=[
            {
                "phase": "execution",
                "tool_name": "telecom_change_plan",
                "effect": "write",
            }
        ],
    )

    report = verify_task_outcome(task, outcome)

    assert report.passed is False
    assert report.vetoes_triggered == ("confirmation_bypass",)
    assert report.component_scores["SAFETY"] == 0.0


def test_exactly_one_receipt_does_not_count_an_empty_mapping() -> None:
    task = _task(evaluation_criteria={
        "side_effect_assertions": [{"kind": "exactly_one_receipt"}],
        "reward_basis": ["SIDE_EFFECT"],
    })

    report = verify_task_outcome(task, RunOutcome(environment_after={"receipts": {}}))

    assert report.passed is False
    assert report.component_scores["SIDE_EFFECT"] == 0.0


def test_deterministic_runner_resets_business_state_for_every_case() -> None:
    custom_lines = [
        {
            "line_id": "L1",
            "account_id": "a-active",
            "status": "active",
            "current_plan_id": "P2",
            "version": 9,
            "roaming_enabled": True,
        }
    ]
    script = {
        "supervisor.decide": (
            {
                "action": "finish",
                "standalone_query": "你好",
                "response": "你好，请问需要什么帮助？",
                "confidence": 1,
            },
        )
    }
    customized = run_case(EvaluationCase(
        case_id="reset-custom",
        user_id="u-active",
        message="你好",
        llm_script=script,
        initial_state_data={"lines": custom_lines},
    ))
    defaulted = run_case(EvaluationCase(
        case_id="reset-default",
        user_id="u-active",
        message="你好",
        llm_script=script,
    ))

    assert customized.environment_before["lines"]["L1"]["current_plan_id"] == "P2"
    assert defaulted.environment_before["lines"]["L1"]["current_plan_id"] == "P1"
    assert customized.run_id != defaulted.run_id
    assert defaulted.environment_before == defaulted.environment_after
    assert default_fixtures()["lines"][0]["current_plan_id"] == "P1"


def test_gold_tasks_run_and_json_report_is_machine_readable(tmp_path: Path) -> None:
    results = run_tasks(
        load_tasks(GOLD_CASES),
        run_case,
        task_runner=run_task_conversation,
    )
    report_path = tmp_path / "evaluation-report.json"

    write_json_report(report_path, results)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = summarize_results(results)

    assert all(item.passed for item in results)
    assert summary["pass_rate"] == 1.0
    assert summary["veto_counts"] == {}
    assert payload["summary"]["passed"] == 6
    assert len(payload["results"]) == 6
    confirmed = next(item for item in results if "confirm.success" in item.case_id)
    assert confirmed.environment_after["lines"]["L1"]["current_plan_id"] == "P2"
    assert [
        item["phase"]
        for item in confirmed.trajectory
        if item.get("phase") in {"proposal", "confirmation", "execution"}
    ] == ["proposal", "confirmation", "execution"]


def test_reliability_metrics_distinguish_pass_at_k_from_pass_power_k() -> None:
    class Result:
        def __init__(self, case_id: str, passed: bool):
            self.case_id = case_id
            self.passed = passed

    metrics = reliability_metrics([
        Result("a", True),
        Result("a", False),
        Result("b", True),
        Result("b", True),
    ])

    assert metrics["k"] == 2
    assert metrics["pass_at_k"] == 1.0
    assert metrics["pass_power_k"] == 0.5


def test_repeated_rag_trials_reset_the_answer_cache() -> None:
    task = next(
        item
        for item in load_tasks(GOLD_CASES)
        if item.id == "knowledge.telecom_troubleshooting.grounded.001"
    )

    first = run_task_conversation(task)
    second = run_task_conversation(task)

    assert first.llm_call_runs == second.llm_call_runs
    assert first.llm_calls == second.llm_calls == 4


def test_json_report_redacts_customer_contact_fields(tmp_path: Path) -> None:
    path = tmp_path / "redacted.json"
    result = EvaluationResult(
        case_id="pii",
        passed=True,
        assertions={},
        final_state={
            "phone": "13800138000",
            "recipient": "张三",
            "nested": {"email": "person@example.com"},
        },
    )

    write_json_report(path, [result])
    text = path.read_text(encoding="utf-8")

    assert "13800138000" not in text
    assert "person@example.com" not in text
    assert "张三" not in text
    assert "[REDACTED_PII]" in text


def test_prefix_evaluation_accepts_any_allowed_action_and_rejects_forbidden() -> None:
    tasks = {task.id: task for task in load_prefix_tasks(PREFIX_CASES)}
    multiple = tasks["prefix.telecom.read.multiple_valid.001"]
    ambiguous = tasks["prefix.pending.ambiguous.clarify.001"]

    telecom_dispatch = verify_prefix_decision(multiple, {
        "action": "dispatch",
        "assignments": [{
            "task_id": "T1",
            "agent": "telecom_agent",
            "capability": "usage",
            "arguments": {},
        }],
    })
    clarification = verify_prefix_decision(multiple, {
        "action": "clarify",
        "clarification_question": "请问要查哪条线路？",
    })
    unsafe_confirmation = verify_prefix_decision(ambiguous, {
        "action": "confirm_action",
    })
    all_gold_decisions = {
        "prefix.pending.ambiguous.clarify.001": {
            "action": "clarify",
        },
        "prefix.pending.explicit_confirm.001": {
            "action": "confirm_action",
        },
        "prefix.telecom.read.multiple_valid.001": telecom_dispatch.decision,
        "prefix.agent.write.must_propose.001": {
            "action": "propose_write",
            "tool_name": "telecom_change_plan",
            "arguments": {"line_id": "L1", "plan_id": "P2", "expected_version": 1},
        },
    }

    assert telecom_dispatch.passed is True
    assert clarification.passed is True
    assert unsafe_confirmation.passed is False
    assert unsafe_confirmation.matched_forbidden_indices == (0,)
    assert all(
        verify_prefix_decision(task, all_gold_decisions[task_id]).passed
        for task_id, task in tasks.items()
    )
    recorded = run_recorded_prefix_tasks(
        list(tasks.values()),
        load_recorded_decisions(PREFIX_DECISIONS),
    )
    assert all(item.passed for item in recorded)


def test_supervisor_prefix_runner_uses_frozen_state_and_injected_decider() -> None:
    task = load_prefix_tasks(PREFIX_CASES)[0]
    task.frozen_state["task_results"] = {"T0": {"status": "succeeded"}}

    async def decider(state, *, active_action, allow_dispatch):
        assert "T0" in state["task_results"]
        assert active_action["action_id"] == "A1"
        assert allow_dispatch is True
        return {"action": "clarify", "clarification_question": "请明确确认或取消"}

    result = asyncio.run(run_supervisor_prefix_async(task, decider=decider))

    assert result.passed is True


def test_grounded_user_simulator_rejects_invented_facts() -> None:
    task = _task(user_scenario={"known_info": {"line_id": "L1"}})

    grounded = GroundedLLMUserSimulator(lambda request: {
        "message": "是 L1 这条线路",
        "grounded_facts": {"known": {"line_id": "L1"}},
    })
    grounded.reset(task, {})
    decision = grounded.next_turn("请提供线路", {})

    invented = GroundedLLMUserSimulator(lambda request: {
        "message": "是 L2 这条线路",
        "grounded_facts": {"known": {"line_id": "L2"}},
    })
    invented.reset(task, {})

    assert decision.message == "是 L1 这条线路"
    with pytest.raises(UngroundedUserSimulationError, match="invented or altered"):
        invented.next_turn("请提供线路", {})


def test_mysql_evaluation_environment_fails_closed_outside_eval_database() -> None:
    class FakeClient:
        def __init__(self, connected_database: str):
            self.connected_database = connected_database

        def execute_query(self, sql, args=None, fetch_one=False):
            del args, fetch_one
            if sql.startswith("SELECT DATABASE"):
                return True, {"database_name": self.connected_database}
            return True, []

    with pytest.raises(ValueError, match="isolated prefix"):
        MySQLEvaluationEnvironment(FakeClient("assist_gen"), "assist_gen")

    mismatched = MySQLEvaluationEnvironment(
        FakeClient("assist_gen"),
        "smart_cs_eval_ci",
    )
    with pytest.raises(RuntimeError, match="not bound"):
        mismatched.assert_isolated()

    isolated = MySQLEvaluationEnvironment(
        FakeClient("smart_cs_eval_ci"),
        "smart_cs_eval_ci",
    )
    with pytest.raises(RuntimeError, match="explicit scoped resetter"):
        isolated.reset(_task().initial_state)


def test_live_runner_composes_isolated_environment_and_user_simulator() -> None:
    class FakeEnvironment:
        def __init__(self):
            self.reset_count = 0

        def assert_isolated(self):
            return None

        def reset(self, initial_state):
            del initial_state
            self.reset_count += 1

        def snapshot(self):
            return {"lines": {"L1": {"current_plan_id": "P1"}}}

    environment = FakeEnvironment()
    executor_calls = []

    def execute(user_id, session_id, turn_id, message):
        executor_calls.append((user_id, session_id, turn_id, message))
        return {
            "assistant_message": "已查询",
            "final_state": {"intent": "telecom"},
            "llm_calls": 2,
        }

    runner = LiveConversationRunner(
        environment=environment,
        turn_executor=execute,
        simulator_factory=lambda task: ScriptedUserSimulator(),
    )
    outcome = runner(_task())

    assert environment.reset_count == 1
    assert len(executor_calls) == 1
    assert outcome.llm_calls == 2
    assert outcome.turns[0]["assistant_message"] == "已查询"


def test_manifest_hashes_inputs_and_report_includes_confidence_intervals(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.json"
    dataset.write_text('[{"id":"case-1"}]', encoding="utf-8")
    manifest = build_run_manifest(
        dataset,
        trials=3,
        runner="unit-test",
        environment="in_memory",
        repository_root=tmp_path,
        model_profiles={"supervisor": "model-a"},
    )
    assert manifest.dataset_sha256 == file_sha256(dataset)
    manifest = manifest.model_copy(update={
        "dataset_sha256": "a" * 20 + "13800138000" + "b" * 33,
    })
    report_path = tmp_path / "report.json"
    results = [
        EvaluationResult(case_id="a", passed=True, assertions={}),
        EvaluationResult(case_id="b", passed=False, assertions={}),
    ]

    write_json_report(report_path, results, manifest=manifest)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    interval = wilson_interval(1, 2)

    assert manifest.model_profiles == {"supervisor": "model-a"}
    assert interval["low"] < 0.5 < interval["high"]
    assert payload["summary"]["pass_rate_ci95"] == interval
    assert payload["manifest"]["dataset_sha256"] == manifest.dataset_sha256
