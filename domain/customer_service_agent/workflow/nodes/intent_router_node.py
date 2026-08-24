from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.llm.llm_service import invoke_llm
from pkg.telemetry import record_json_parse, record_route
from pkg.llm import parse_json_object

INTENT_SYSTEM_PROMPT = """你是一个专业的意图识别Agent，负责分析用户的客服消息。

请从以下维度分析用户意图：
1. 一级意图分类: consultation(咨询), complaint(投诉), transaction(交易办理), account(账户), compliance(合规), chitchat(闲聊问候), unknown(无法识别)
2. 二级意图: 具体的业务子类型
3. 置信度: 0.0-1.0
4. 关键实体: 提取订单号、工单号、产品名、金额等关键信息
5. 建议路由: knowledge_rag(知识查询), ticket_handler(工单处理), compliance_checker(合规审查), react_customer_service(动态客服处理), plan_execute_onboarding(开户流程规划执行), fallback_conversation(闲聊和兜底对话)

只返回JSON：
{
  "primary_intent": "consultation",
  "secondary_intent": "product_inquiry",
  "confidence": 0.95,
  "entities": {"product": "理财产品A"},
  "suggested_agent": "knowledge_rag"
}

规则：
- 涉及资金安全、账户异常、欺诈举报、敏感合规 → compliance_checker
- 涉及“查一下并处理”、“如果需要就创建工单”、“查询后决定下一步”、退款/订单/工单组合诉求 → react_customer_service
- 涉及开户流程、怎么开户、一步步开户指导、开户材料、开户相关风险测评 → plan_execute_onboarding
- 涉及单次退款、理赔、投诉、工单创建或工单查询 → ticket_handler
- 涉及产品咨询、利率查询、政策了解、订单查询 → knowledge_rag
- 涉及问候、感谢、告别、普通闲聊或无法判断具体业务意图 → fallback_conversation
"""


INTENT_SYSTEM_PROMPT += """

开户路由补充规则：完整开户流程、开户准备材料、与开户有关的风险测评、开户资格或限制条件，以及这些事项的组合查询，均路由到 plan_execute_onboarding。
示例：
- “开户需要什么材料” -> plan_execute_onboarding
- “风险测评会影响开户吗” -> plan_execute_onboarding
- “我这种情况能开户吗” -> plan_execute_onboarding
一般风险问题、非开户风险问题，以及创建或查询工单的请求，不得因此路由到 plan_execute_onboarding。
负例：
- “这个投资产品有什么风险” -> knowledge_rag
- “帮我创建退款工单” -> ticket_handler
- “查询工单 TK-1001” -> ticket_handler
"""


class IntentDecision(BaseModel):
    primary_intent: Literal["consultation", "complaint", "transaction", "account", "compliance", "chitchat", "unknown"] = "unknown"
    secondary_intent: str = Field(default="unknown", max_length=128)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    entities: dict[str, str] = Field(default_factory=dict)
    suggested_agent: Literal[
        "knowledge_rag",
        "ticket_handler",
        "compliance_checker",
        "react_customer_service",
        "plan_execute_onboarding",
        "fallback_conversation",
    ] = "fallback_conversation"


def _parse_json(content: str) -> dict:
    parsed = parse_json_object(content)
    if parsed is not None:
        try:
            result = IntentDecision.model_validate(parsed).model_dump()
            record_json_parse("intent.classify", True)
            return result
        except ValidationError:
            pass
    record_json_parse("intent.classify", False)
    return IntentDecision().model_dump()


def intent_router_node(state: ChatState) -> dict:
    query = state.get("fused_query") or state["raw_query"]
    response = invoke_llm([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=f"用户消息：{query}"),
    ], run_name="intent.classify")
    result = _parse_json(str(response.content))
    primary_intent = result.get("primary_intent", "unknown")
    confidence = float(result.get("confidence", 0.0) or 0.0)
    suggested_agent = result.get("suggested_agent") or "fallback_conversation"
    fallback_used = primary_intent in {"unknown", "chitchat"} or confidence < 0.6
    if fallback_used:
        suggested_agent = "fallback_conversation"
    route_valid = True
    record_route(valid=route_valid, fallback_used=fallback_used)
    return {
        "intent": suggested_agent,
        "primary_intent": primary_intent,
        "secondary_intent": result.get("secondary_intent", "unknown"),
        "confidence": confidence,
        "entities": result.get("entities", {}) or {},
        "current_agent": "intent_router",
        "node_logs": [f"识别意图：{suggested_agent}"],
    }
