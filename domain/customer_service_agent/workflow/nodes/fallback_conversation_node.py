from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.llm.llm_service import invoke_llm
from pkg.log.logger import get_logger

logger = get_logger()

FALLBACK_SAFE_RESPONSE = "您好，我可以协助您进行产品咨询、订单查询、工单处理或开户指导。请告诉我您想办理的业务。"

FALLBACK_CONVERSATION_SYSTEM_PROMPT = """你是一个友好、专业的客服兜底对话 Agent。
你的任务是回应问候、感谢、告别、普通闲聊，以及暂时无法识别业务意图的消息。

回复要求：
1. 使用自然、简洁的中文回复，先回应用户当前表达。
2. 如果用户只是简单问候，保持简短友好，不要堆砌所有业务入口。
3. 如果请求含糊，说明继续处理需要用户补充什么信息。
4. 根据对话内容，自然引导到产品咨询、订单查询、工单处理或开户指导中的相关能力。
5. 不强行营销，不虚构产品、订单、工单、账户或政策事实。
6. 不声称已经查询、创建或办理任何实际业务。
只输出给用户的最终回复，不要输出分析过程或 JSON。
"""


def fallback_conversation_node(state: ChatState) -> dict:
    query = state.get("fused_query") or state["raw_query"]
    raw_query = state["raw_query"]
    try:
        response = invoke_llm(
            [
                SystemMessage(content=FALLBACK_CONVERSATION_SYSTEM_PROMPT),
                HumanMessage(content=f"用户当前消息：{raw_query}\n融合后的对话请求：{query}"),
            ],
            run_name="fallback_conversation.reply",
        )
        answer = str(response.content).strip() or FALLBACK_SAFE_RESPONSE
        node_log = "兜底对话回复已生成"
    except Exception:
        logger.exception("Fallback conversation LLM invocation failed")
        answer = FALLBACK_SAFE_RESPONSE
        node_log = "兜底对话模型调用失败，已使用安全话术"

    return {
        "sub_results": {**state.get("sub_results", {}), "fallback_conversation": answer},
        "current_agent": "fallback_conversation",
        "node_logs": [node_log],
    }
