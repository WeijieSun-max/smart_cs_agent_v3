from __future__ import annotations

from typing import Any, TypedDict

from langgraph.constants import START
from langgraph.graph import END, StateGraph

from domain.customer_service_agent.orchestration.bounded_react import execute_bounded_react
from domain.customer_service_agent.orchestration.models import AgentResult, QueryUnderstandingResult, TaskSpec
from domain.customer_service_agent.orchestration.resource_resolver import needs_order_resolution, resolve_order
from domain.customer_service_agent.orchestration.skill_executor import execute_plan_recommendation
from domain.customer_service_agent.orchestration.tool_scheduler import execute_read_task
from domain.customer_service_agent.orchestration.write_proposal import prepare_write_proposal
from domain.customer_service_agent.retrieval import retrieve_grounded_answer
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.identity import RequestIdentityContext


class DomainAgentState(TypedDict):
    parent_state: ChatState
    identity: RequestIdentityContext
    tasks: list[dict[str, Any]]
    results: list[dict[str, Any]]


async def _telecom_executor(state: DomainAgentState) -> dict[str, Any]:
    parent = state["parent_state"]
    identity = state["identity"]
    results = []
    for task_data in state["tasks"]:
        task = TaskSpec.model_validate(task_data)
        if task.execution_mode == "react":
            results.append((await execute_bounded_react(
                task,
                parent.get("normalized_query") or parent["raw_query"],
                identity,
            )).model_dump(mode="json"))
        elif task.capability == "plan_recommendation":
            update = await execute_plan_recommendation(parent, identity)
            results.append(AgentResult(
                task_id=task.task_id,
                status="succeeded",
                facts={
                    "skill_selection": update.get("skill_selection"),
                    "skill_result": update.get("skill_result"),
                },
                user_fragment=_update_text(update),
            ).model_dump(mode="json"))
        elif task.capability == "telecom_troubleshooting":
            update = await retrieve_grounded_answer(parent, "telecom", task.capability)
            results.append(AgentResult(
                task_id=task.task_id,
                status="succeeded",
                facts=update.get("task_results", {}),
                user_fragment=_update_text(update),
            ).model_dump(mode="json"))
        elif task.effect == "read":
            results.append((await execute_read_task(task, identity)).model_dump(mode="json"))
        else:
            text = await prepare_write_proposal(
                parent.get("normalized_query") or parent["raw_query"],
                task.capability,
                identity,
                resolved_entities={key: str(value) for key, value in task.arguments.items()},
            )
            results.append(_proposal_result(task, text).model_dump(mode="json"))
    return {"results": results}


async def _retail_executor(state: DomainAgentState) -> dict[str, Any]:
    parent = state["parent_state"]
    identity = state["identity"]
    understanding = QueryUnderstandingResult.model_validate(parent["query_understanding"])
    results = []
    for task_data in state["tasks"]:
        task = TaskSpec.model_validate(task_data)
        resolution = None
        if needs_order_resolution(task.capability, understanding):
            resolution = await resolve_order(understanding, identity)
            if resolution.status != "resolved":
                results.append(AgentResult(
                    task_id=task.task_id,
                    status="failed",
                    facts={"order_resolution": resolution.model_dump(mode="json")},
                    user_fragment=resolution.user_fragment,
                    error_code=f"entity.order_{resolution.status}",
                ).model_dump(mode="json"))
                continue
            task = task.model_copy(update={
                "arguments": {**task.arguments, "order_id": str(resolution.order_id)},
            })
        if task.execution_mode == "react":
            result = await execute_bounded_react(
                task,
                parent.get("normalized_query") or parent["raw_query"],
                identity,
            )
        elif task.capability == "retail_policy":
            update = await retrieve_grounded_answer(parent, "retail", task.capability)
            result = AgentResult(
                task_id=task.task_id,
                status="succeeded",
                facts=update.get("task_results", {}),
                user_fragment=_update_text(update),
            )
        elif task.effect == "read":
            result = await execute_read_task(task, identity)
        else:
            text = await prepare_write_proposal(
                parent.get("normalized_query") or parent["raw_query"],
                task.capability,
                identity,
                resolved_entities={key: str(value) for key, value in task.arguments.items()},
            )
            result = _proposal_result(task, text)
        if resolution is not None:
            result = result.model_copy(update={
                "facts": {**result.facts, "order_resolution": resolution.model_dump(mode="json")},
            })
        results.append(result.model_dump(mode="json"))
    return {"results": results}


def _proposal_result(task: TaskSpec, text: str | None) -> AgentResult:
    fragment = text or "当前操作暂不可用。"
    return AgentResult(
        task_id=task.task_id,
        status="needs_confirmation" if fragment.startswith("待确认：") else "succeeded",
        user_fragment=fragment,
    )


def _update_text(update: dict[str, Any]) -> str:
    values = update.get("sub_results", {}).values()
    return "\n\n".join(value for value in values if isinstance(value, str))


def _build_subgraph(executor) -> Any:
    graph = StateGraph(DomainAgentState)
    graph.add_node("execute", executor)
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    return graph.compile()


telecom_agent_graph = _build_subgraph(_telecom_executor)
retail_agent_graph = _build_subgraph(_retail_executor)
