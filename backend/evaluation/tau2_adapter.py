from __future__ import annotations

from typing import Any

from evaluation.harness import EvaluationCase, task_to_case
from evaluation.schema import (
    EvaluationBudgets,
    EvaluationCriteria,
    EvaluationTask,
    ForbiddenCriteria,
    InitialState,
    ReferenceAction,
    RewardComponent,
    UserScenario,
)


def import_tau2_task(
    task: dict[str, Any],
    *,
    domain: str = "shared",
    capability: str = "fallback",
) -> EvaluationTask:
    """Import the current τ-bench task shape without claiming domain compatibility.

    DB hashes and callable environment assertions belong to the τ-bench domain and
    cannot be executed against Smart CS until a domain-specific state mapper is
    supplied.  They are preserved as ``external_assertions`` and source metadata.
    Reference actions remain diagnostic unless ACTION explicitly gates the source task.
    """

    criteria = task.get("evaluation_criteria") or {}
    scenario = task.get("user_scenario") or {}
    instructions = scenario.get("instructions") or scenario
    known_info = instructions.get("known_info") or {}
    user_id = str(
        task.get("user_id")
        or (known_info.get("user_id") if isinstance(known_info, dict) else "")
        or "eval-user"
    )

    source_basis = tuple(str(value) for value in criteria.get("reward_basis") or ())
    supported_basis: list[RewardComponent] = [RewardComponent.SAFETY]
    if "COMMUNICATE" in source_basis:
        supported_basis.append(RewardComponent.COMMUNICATE)
    if "NL_ASSERTION" in source_basis:
        supported_basis.append(RewardComponent.NL_ASSERTION)
    if "ACTION" in source_basis:
        supported_basis.append(RewardComponent.ACTION)

    actions = tuple(
        ReferenceAction.model_validate(item)
        for item in criteria.get("actions") or ()
        if isinstance(item, dict) and item.get("name")
    )
    external_assertions = tuple(
        item
        for item in criteria.get("env_assertions") or ()
        if isinstance(item, dict)
    )
    if "DB" in source_basis:
        external_assertions = (
            {"kind": "tau2_db_hash", "requires_domain_state_mapper": True},
            *external_assertions,
        )

    mapped_domain = domain if domain in {"telecom", "retail", "composite", "shared"} else "shared"
    return EvaluationTask(
        id=str(task.get("id") or task.get("task_id")),
        split="dev",
        domain=mapped_domain,
        capability=capability,
        user_id=user_id,
        ticket=str(task.get("ticket") or task.get("instruction") or task.get("message") or ""),
        expected_route=None if mapped_domain == "shared" else mapped_domain,
        expected_capabilities=(capability,),
        user_scenario=UserScenario(
            known_info=known_info,
            unknown_info=instructions.get("unknown_info"),
            task_instructions=instructions.get("task_instructions"),
        ),
        initial_state=InitialState(
            fixture=f"tau2:{mapped_domain}",
            initialization_actions=tuple(
                (task.get("initial_state") or {}).get("initialization_actions") or ()
            ),
        ),
        evaluation_criteria=EvaluationCriteria(
            communicate={
                "all": tuple(str(value) for value in criteria.get("communicate_info") or ())
            },
            nl_assertions=tuple(str(value) for value in criteria.get("nl_assertions") or ()),
            forbidden=ForbiddenCriteria(),
            budgets=EvaluationBudgets(),
            reward_basis=tuple(dict.fromkeys(supported_basis)),
            external_assertions=external_assertions,
        ),
        reference_actions=actions,
        metadata={
            "source": "tau-bench",
            "source_reward_basis": source_basis,
            "requires_domain_state_mapper": bool(
                {"DB", "ENV_ASSERTION"}.intersection(source_basis)
            ),
        },
    )


def from_tau2_task(task: dict[str, Any]) -> EvaluationCase:
    """Backward-compatible single-turn adapter used by legacy callers.

    It intentionally does not turn τ-bench's reference action list into an exact
    ``expected_tool_sequence``.
    """

    return task_to_case(import_tau2_task(task))
