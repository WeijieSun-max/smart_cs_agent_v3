"""组装带持久化 checkpoint 和不可绕过合规节点的客服工作流。"""

from __future__ import annotations

from typing import Any

from langgraph.constants import START
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda

from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.customer_service_agent.workflow.nodes.compliance_checker_node import compliance_checker_node, compliance_checker_node_async
from domain.customer_service_agent.workflow.nodes.history_fusion_node import history_fusion_node
from domain.customer_service_agent.workflow.nodes.response_synthesizer_node import response_synthesizer_node
from domain.customer_service_agent.workflow.nodes.response_writer_node import response_writer_node, response_writer_node_async
from domain.customer_service_agent.workflow.nodes.supervisor_graph_nodes import (
    domain_dispatch_node,
    domain_dispatch_node_sync,
    dispatch_route,
    pending_action_execution_node,
    pending_action_execution_node_sync,
    supervisor_manager_node,
    supervisor_manager_node_sync,
    supervisor_route,
)
from domain.shared.checkpoint import checkpoint_saver_service
from pkg.log.logger import get_logger

logger=get_logger(); _workflow: Any | None=None


def initialize_workflow() -> None:
    """编译进程级工作流，并报告工具能力契约漂移。"""

    global _workflow
    _workflow=_build_customer_service_workflow()
    _report_capability_contracts()
    logger.info("LLM manager workflow initialized with knowledge/telecom/retail agents")


def _report_capability_contracts() -> None:
    """记录能力元数据问题；启动后仍由各执行边界继续强制校验。"""

    from domain.customer_service_agent.orchestration.capability_index import validate_capability_contracts

    problems = validate_capability_contracts()
    if problems:
        logger.error("Capability contract drift detected: %s", "; ".join(problems))
    else:
        logger.info("Capability contract validated: vocabulary derived from tool registry is consistent")


def get_workflow() -> Any:
    """返回已编译图；依赖尚未完成时立即失败。"""

    if _workflow is None: raise RuntimeError("Customer service workflow has not been initialized")
    return _workflow


def _build_customer_service_workflow() -> Any:
    """构建主图并确保所有响应路径都经过 writer、compliance 和 synthesizer。

    Supervisor 与领域 Agent 可有界往返；治理动作单独进入确定性执行节点。
    无论哪条分支结束，草稿都必须先经过合规检查，之后才能写入最终响应。
    """

    workflow=StateGraph(ChatState)
    workflow.add_node("history_fusion_node",history_fusion_node)
    workflow.add_node("supervisor_manager_node",RunnableLambda(supervisor_manager_node_sync,afunc=supervisor_manager_node))
    workflow.add_node("domain_dispatch_node",RunnableLambda(domain_dispatch_node_sync,afunc=domain_dispatch_node))
    workflow.add_node("pending_action_execution_node",RunnableLambda(pending_action_execution_node_sync,afunc=pending_action_execution_node))
    workflow.add_node("response_writer_node",RunnableLambda(response_writer_node,afunc=response_writer_node_async))
    workflow.add_node("compliance_checker_node",RunnableLambda(compliance_checker_node,afunc=compliance_checker_node_async))
    workflow.add_node("response_synthesizer_node",response_synthesizer_node)
    workflow.add_edge(START,"history_fusion_node")
    workflow.add_edge("history_fusion_node","supervisor_manager_node")
    workflow.add_conditional_edges(
        "supervisor_manager_node",
        supervisor_route,
        {
            "dispatch":"domain_dispatch_node",
            "action":"pending_action_execution_node",
            "respond":"response_writer_node",
        },
    )
    workflow.add_conditional_edges(
        "domain_dispatch_node",
        dispatch_route,
        {"review":"supervisor_manager_node","respond":"response_writer_node"},
    )
    workflow.add_edge("pending_action_execution_node","response_writer_node")
    workflow.add_edge("response_writer_node","compliance_checker_node")
    workflow.add_edge("compliance_checker_node","response_synthesizer_node")
    workflow.add_edge("response_synthesizer_node",END)
    return workflow.compile(checkpointer=checkpoint_saver_service.instance.get_checkpointer())
