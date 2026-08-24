from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState


def merge_node_logs(existing: list[str] | None, new: list[str]) -> list[str]:
    """Append node logs, or clear them when a node emits the reset marker."""
    if new == ["RESET"]:
        return []
    return [*(existing or []), *new]


class ChatState(MessagesState):
    user_id: str | None
    session_id: str
    current_time: str
    turn_id: str
    raw_query: str
    prior_context: str
    memory_packet: dict[str, Any] | None
    user_message_persisted: bool
    context_text: str
    intent: str | None
    primary_intent: str | None
    secondary_intent: str | None
    confidence: float | None
    entities: dict[str, str]
    sub_results: dict[str, Any]
    compliance_passed: bool
    compliance_result: dict[str, Any]
    final_response: str
    current_agent: str
    node_logs: Annotated[list[str], merge_node_logs]
    pending_action_intent: str
    pending_action_route: str
    skill_selection: dict[str, Any] | None
    skill_result: dict[str, Any] | None
    request_id: str | None
    identity_source: str
    auth_strength: str
    route_decision: dict[str, Any] | None
    task_plan: dict[str, Any] | None
    task_results: dict[str, Any]


def create_chat_state(
    user_id: str | None,
    session_id: str | None,
    query: str,
    *,
    turn_id: str = "untracked",
    prior_context: str = "",
    memory_packet: dict[str, Any] | None = None,
    user_message_persisted: bool = False,
) -> ChatState:
    resolved_session_id = session_id or uuid.uuid4().hex
    return {
        "user_id": user_id,
        "session_id": resolved_session_id,
        "current_time": datetime.now().strftime("%Y-%m-%d"),
        "turn_id": turn_id,
        "raw_query": query,
        "prior_context": prior_context,
        "memory_packet": memory_packet,
        "user_message_persisted": user_message_persisted,
        "messages": [HumanMessage(content=query)],
        "context_text": "",
        "intent": None,
        "primary_intent": None,
        "secondary_intent": None,
        "confidence": None,
        "entities": {},
        "sub_results": {},
        "compliance_passed": True,
        "compliance_result": {},
        "final_response": "",
        "current_agent": "",
        "node_logs": [],
        "pending_action_intent": "none",
        "pending_action_route": "route",
        "skill_selection": None,
        "skill_result": None,
        "request_id": None,
        "identity_source": "request_body",
        "auth_strength": "unverified_frontend",
        "route_decision": None,
        "task_plan": None,
        "task_results": {},
    }
