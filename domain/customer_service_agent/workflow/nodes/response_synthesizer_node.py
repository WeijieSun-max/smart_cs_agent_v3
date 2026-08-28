"""在合规结论之后产生唯一最终文本并追加助手消息。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from domain.customer_service_agent.workflow.entity.chat_state import ChatState


def response_synthesizer_node(state: ChatState) -> dict:
    """合规失败时替换为人工处理提示，否则选择已审查草稿或安全兜底。"""

    if not state.get("compliance_passed", True):
        final_response = "抱歉，您的请求或回复内容涉及敏感信息，已转交人工客服处理。"
    else:
        draft = state.get("draft_response") or ""
        parts = [value for value in state.get("sub_results", {}).values() if isinstance(value, str)]
        final_response = draft if draft.strip() else "\n\n".join(parts)
        if not final_response:
            final_response = "抱歉，暂时无法处理您的请求，请稍后重试。"
    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
        "current_agent": "response_synthesizer",
        "node_logs": ["已生成最终回复"],
    }
