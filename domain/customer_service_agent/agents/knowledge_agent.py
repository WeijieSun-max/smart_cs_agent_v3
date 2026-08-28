"""只负责非结构化知识检索的 Knowledge Agent 执行器。"""

from __future__ import annotations

from typing import Any

from domain.customer_service_agent.orchestration.models import AgentAssignment, AgentResult
from domain.customer_service_agent.retrieval import retrieve_grounded_answer
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from pkg.telemetry import normalize_error

_CAPABILITY_DOMAIN = {
    "telecom_troubleshooting": "telecom",
    "retail_policy": "retail",
}


async def run_knowledge_agent(
    assignment: AgentAssignment,
    state: ChatState,
) -> AgentResult:
    """执行一次有来源约束的知识任务并转换为统一 AgentResult。

    capability 决定检索域，用户原始问题优先于 Supervisor 改写目标。检索器
    异常会转成安全错误结果，避免基础设施异常越过最终响应合成与合规节点。
    """

    domain = _CAPABILITY_DOMAIN.get(assignment.capability)
    if domain is None:
        return AgentResult(
            task_id=assignment.task_id,
            agent="knowledge_agent",
            status="failed",
            user_fragment="该知识任务不在当前知识库能力范围内。",
            error_code="knowledge.capability_not_supported",
        )
    # 单任务时优先使用 Supervisor 已完成指代消解的 standalone query；复合请求
    # 中则只使用本 assignment，防止把兄弟领域问题带入知识检索和回答提示。
    dispatched_assignments = state.get("agent_assignments") or []
    retrieval_query = (
        assignment.objective
        if len(dispatched_assignments) > 1
        else state.get("normalized_query") or assignment.objective
    )
    child_state: ChatState = {
        **state,
        "raw_query": retrieval_query,
        "normalized_query": retrieval_query,
    }
    try:
        update = await retrieve_grounded_answer(
            child_state,
            domain,
            assignment.capability,
        )
    except Exception as exc:
        error = normalize_error(exc)
        return AgentResult(
            task_id=assignment.task_id,
            agent="knowledge_agent",
            status="failed",
            user_fragment="知识库暂时不可用，建议稍后重试或转人工客服。",
            error_code=str(error["error_code"]),
        )
    fragments = [
        value
        for value in (update.get("sub_results") or {}).values()
        if isinstance(value, str) and value
    ]
    facts: dict[str, Any] = dict(update.get("task_results") or {})
    return AgentResult(
        task_id=assignment.task_id,
        agent="knowledge_agent",
        status="succeeded",
        facts=facts,
        user_fragment="\n\n".join(fragments) or "知识库中没有找到可靠依据。",
    )
