from __future__ import annotations

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
from evaluation.reporting import reliability_metrics, summarize_results, write_json_report
from evaluation.schema import EvaluationTask, load_tasks
from evaluation.tau2_adapter import from_tau2_task, import_tau2_task
from evaluation.verifiers import verify_task_outcome


GOLD_CASES = Path(__file__).parents[1] / "evaluation" / "datasets" / "dev" / "gold_cases.json"


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
