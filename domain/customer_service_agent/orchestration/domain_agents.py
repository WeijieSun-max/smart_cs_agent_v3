from __future__ import annotations

from typing import Any, TypedDict

from langgraph.constants import START
from langgraph.graph import END, StateGraph

from domain.customer_service_agent.agents import run_knowledge_agent, run_tool_agent
from domain.customer_service_agent.orchestration.models import AgentAssignment, AgentResult
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.identity import RequestIdentityContext


class DomainAgentState(TypedDict):
    parent_state: ChatState
    identity: RequestIdentityContext
    assignment: dict[str, Any]
    result: dict[str, Any] | None


async def _knowledge_executor(state: DomainAgentState) -> dict[str, Any]:
    assignment = AgentAssignment.model_validate(state["assignment"])
    result = await run_knowledge_agent(assignment, state["parent_state"])
    return {"result": result.model_dump(mode="json")}


async def _telecom_executor(state: DomainAgentState) -> dict[str, Any]:
    return await _tool_executor(state, "telecom_agent")


async def _retail_executor(state: DomainAgentState) -> dict[str, Any]:
    return await _tool_executor(state, "retail_agent")


async def _tool_executor(state: DomainAgentState, expected_agent: str) -> dict[str, Any]:
    assignment = AgentAssignment.model_validate(state["assignment"])
    if assignment.agent != expected_agent:
        result = AgentResult(
            task_id=assignment.task_id,
            agent=assignment.agent,
            status="failed",
            user_fragment="Supervisor 分配的 Agent 类型与子图不一致。",
            error_code="supervisor.agent_mismatch",
        )
    else:
        result = await run_tool_agent(
            assignment,
            state["parent_state"],
            state["identity"],
        )
    return {"result": result.model_dump(mode="json")}


def _build_subgraph(executor) -> Any:
    graph = StateGraph(DomainAgentState)
    graph.add_node("execute", executor)
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    return graph.compile()


knowledge_agent_graph = _build_subgraph(_knowledge_executor)
telecom_agent_graph = _build_subgraph(_telecom_executor)
retail_agent_graph = _build_subgraph(_retail_executor)
