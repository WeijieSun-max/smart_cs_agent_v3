from __future__ import annotations

from domain.customer_service_agent.memory.conversation_context import (
    build_conversation_context,
    has_conversation_context,
    normalize_conversation_context,
)
from domain.customer_service_agent.memory.models import MemoryPacket
from domain.customer_service_agent.workflow.entity.chat_state import ChatState


def history_fusion_node(state: ChatState) -> dict:
    """Build the structured LLM context without replaying historical chat messages."""

    packet_data = state.get("memory_packet")
    if packet_data:
        context = build_conversation_context(MemoryPacket.model_validate(packet_data))
        source = "layered"
    else:
        context = normalize_conversation_context(state.get("conversation_context"))
        source = "recent"

    has_context = has_conversation_context(context)
    return {
        "conversation_context": context.model_dump(mode="json"),
        "current_agent": "history_fusion",
        "node_logs": [
            "已融合结构化分层记忆上下文"
            if source == "layered"
            else (
                "已融合结构化近期会话上下文"
                if has_context
                else "当前会话暂无历史上下文"
            )
        ],
    }
