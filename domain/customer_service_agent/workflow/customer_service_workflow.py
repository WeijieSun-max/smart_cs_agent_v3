from __future__ import annotations

from typing import Any

from langgraph.constants import START
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda

from domain.customer_service_agent.orchestration.query_understanding import query_understanding_node
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.customer_service_agent.workflow.nodes.compliance_checker_node import compliance_checker_node
from domain.customer_service_agent.workflow.nodes.history_fusion_node import history_fusion_node
from domain.customer_service_agent.workflow.nodes.response_synthesizer_node import response_synthesizer_node
from domain.customer_service_agent.workflow.nodes.response_writer_node import response_writer_node
from domain.customer_service_agent.workflow.nodes.supervisor_graph_nodes import (
    domain_dispatch_node,
    domain_dispatch_node_sync,
    pending_action_gate_node,
    pending_action_gate_node_sync,
    pending_action_route,
    result_aggregator_node,
    supervisor_planner_node,
    supervisor_router_node,
)
from domain.shared.checkpoint import checkpoint_saver_service
from pkg.log.logger import get_logger

logger=get_logger(); _workflow: Any | None=None


def initialize_workflow() -> None:
    global _workflow
    _workflow=_build_customer_service_workflow()
    _report_capability_contracts()
    logger.info("Telecom/Retail supervisor workflow initialized")


def _report_capability_contracts() -> None:
    from domain.customer_service_agent.orchestration.capability_index import validate_capability_contracts

    problems = validate_capability_contracts()
    if problems:
        logger.error("Capability contract drift detected: %s", "; ".join(problems))
    else:
        logger.info("Capability contract validated: vocabulary derived from tool registry is consistent")


def get_workflow() -> Any:
    if _workflow is None: raise RuntimeError("Customer service workflow has not been initialized")
    return _workflow


def _build_customer_service_workflow() -> Any:
    workflow=StateGraph(ChatState)
    workflow.add_node("history_fusion_node",history_fusion_node)
    workflow.add_node("pending_action_gate_node",RunnableLambda(pending_action_gate_node_sync,afunc=pending_action_gate_node))
    workflow.add_node("query_understanding_node",query_understanding_node)
    workflow.add_node("supervisor_router_node",supervisor_router_node)
    workflow.add_node("supervisor_planner_node",supervisor_planner_node)
    workflow.add_node("domain_dispatch_node",RunnableLambda(domain_dispatch_node_sync,afunc=domain_dispatch_node))
    workflow.add_node("result_aggregator_node",result_aggregator_node)
    workflow.add_node("response_writer_node",response_writer_node)
    workflow.add_node("compliance_checker_node",compliance_checker_node)
    workflow.add_node("response_synthesizer_node",response_synthesizer_node)
    workflow.add_edge(START,"history_fusion_node")
    workflow.add_edge("history_fusion_node","pending_action_gate_node")
    workflow.add_conditional_edges(
        "pending_action_gate_node",
        pending_action_route,
        {"handled":"compliance_checker_node","continue":"query_understanding_node"},
    )
    workflow.add_edge("query_understanding_node","supervisor_router_node")
    workflow.add_edge("supervisor_router_node","supervisor_planner_node")
    workflow.add_edge("supervisor_planner_node","domain_dispatch_node")
    workflow.add_edge("domain_dispatch_node","result_aggregator_node")
    workflow.add_edge("result_aggregator_node","response_writer_node")
    workflow.add_edge("response_writer_node","compliance_checker_node")
    workflow.add_edge("compliance_checker_node","response_synthesizer_node")
    workflow.add_edge("response_synthesizer_node",END)
    return workflow.compile(checkpointer=checkpoint_saver_service.instance.get_checkpointer())
