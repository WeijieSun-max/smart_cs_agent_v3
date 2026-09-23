"""把领域 Agent 执行器包装为独立、可组合的 LangGraph 子图。"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.constants import START
from langgraph.graph import END, StateGraph

from domain.customer_service_agent.agents import run_knowledge_agent, run_tool_agent
from domain.customer_service_agent.orchestration.models import AgentAssignment, AgentResult
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.identity import RequestIdentityContext


class DomainAgentState(TypedDict):
    """领域子图的最小输入输出状态。

    `parent_state` 只供本次任务读取；子图只返回序列化的 `result`，从而避免
    并行 Agent 直接覆盖父图中的共享字段。
    """

    parent_state: ChatState
    identity: RequestIdentityContext
    assignment: dict[str, Any]
    dependency_results: dict[str, dict[str, Any]]
    result: dict[str, Any] | None


async def _knowledge_executor(state: DomainAgentState) -> dict[str, Any]:
    """校验 assignment 后调用只读知识 Agent。"""

    assignment = AgentAssignment.model_validate(state["assignment"])
    dependency_results = state.get("dependency_results") or {}
    if dependency_results:
        result = await run_knowledge_agent(
            assignment,
            state["parent_state"],
            dependency_results=dependency_results,
        )
    else:
        result = await run_knowledge_agent(assignment, state["parent_state"])
    return {"result": result.model_dump(mode="json")}


async def _telecom_executor(state: DomainAgentState) -> dict[str, Any]:
    """在电信 Agent 身份约束下执行结构化工具任务。"""

    return await _tool_executor(state, "telecom_agent")


async def _retail_executor(state: DomainAgentState) -> dict[str, Any]:
    """在零售 Agent 身份约束下执行结构化工具任务。"""

    return await _tool_executor(state, "retail_agent")


async def _tool_executor(state: DomainAgentState, expected_agent: str) -> dict[str, Any]:
    """拒绝跨领域误分配，再把任务交给通用工具 Agent。"""

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
        dependency_results = state.get("dependency_results") or {}
        if dependency_results:
            result = await run_tool_agent(
                assignment,
                state["parent_state"],
                state["identity"],
                dependency_results=dependency_results,
            )
        else:
            result = await run_tool_agent(
                assignment,
                state["parent_state"],
                state["identity"],
            )
    return {"result": result.model_dump(mode="json")}


def _build_subgraph(executor) -> Any:
    """构建只有一个受控执行节点的领域子图。"""

    graph = StateGraph(DomainAgentState)
    graph.add_node("execute", executor)
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    return graph.compile()


knowledge_agent_graph = _build_subgraph(_knowledge_executor)
telecom_agent_graph = _build_subgraph(_telecom_executor)
retail_agent_graph = _build_subgraph(_retail_executor)
