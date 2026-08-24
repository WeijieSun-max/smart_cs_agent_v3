from __future__ import annotations

from typing import Any

from langgraph.constants import START
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda

from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.customer_service_agent.workflow.nodes.compliance_checker_node import compliance_checker_node
from domain.customer_service_agent.workflow.nodes.history_fusion_node import history_fusion_node
from domain.customer_service_agent.workflow.nodes.response_synthesizer_node import response_synthesizer_node
from domain.customer_service_agent.workflow.nodes.supervisor_node import supervisor_node,supervisor_node_sync
from domain.shared.checkpoint import checkpoint_saver_service
from pkg.log.logger import get_logger

logger=get_logger(); _workflow: Any | None=None


def initialize_workflow() -> None:
    global _workflow
    _workflow=_build_customer_service_workflow()
    logger.info("Telecom/Retail supervisor workflow initialized")


def get_workflow() -> Any:
    if _workflow is None: raise RuntimeError("Customer service workflow has not been initialized")
    return _workflow


def _build_customer_service_workflow() -> Any:
    workflow=StateGraph(ChatState)
    workflow.add_node("history_fusion_node",history_fusion_node)
    workflow.add_node("supervisor_node",RunnableLambda(supervisor_node_sync,afunc=supervisor_node))
    workflow.add_node("compliance_checker_node",compliance_checker_node)
    workflow.add_node("response_synthesizer_node",response_synthesizer_node)
    workflow.add_edge(START,"history_fusion_node")
    workflow.add_edge("history_fusion_node","supervisor_node")
    workflow.add_edge("supervisor_node","compliance_checker_node")
    workflow.add_edge("compliance_checker_node","response_synthesizer_node")
    workflow.add_edge("response_synthesizer_node",END)
    return workflow.compile(checkpointer=checkpoint_saver_service.instance.get_checkpointer())
