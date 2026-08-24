from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, StrictBool, ValidationError

from domain.customer_service_agent.service import knowledge_service, order_service, risk_service, ticket_service
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.llm.llm_service import invoke_llm
from pkg.telemetry import record_json_parse
from pkg.llm import parse_json_object

REACT_SYSTEM_PROMPT = """你是一个客服问题动态处理 ReAct Agent。
你需要观察用户问题，决定是否查询知识库、订单、工单、创建工单或进行风险检查。
请只返回JSON：
{
  "steps": ["将要执行的动作"],
  "knowledge_query": "可选，知识库检索词",
  "need_order_query": false,
  "order_id": "可选",
  "need_ticket_query": false,
  "ticket_id": "可选",
  "need_ticket_create": false,
  "ticket_type": "refund|claim|account_open|account_change|complaint|general",
  "priority": "low|medium|high|urgent",
  "summary": "可选，工单摘要",
  "details": "可选，工单详情",
  "need_risk_check": false,
  "risk_action": "可选，风险动作",
  "amount": 0,
  "final_answer": "基于计划给用户的简短说明"
}
规则：
- 用户要求“查一下并处理”“如果需要就创建工单”“查询后决定下一步”时，按需组合多个动作。
- 没有明确订单号或工单号时，不要编造编号。
- 缺少必要信息时，在 final_answer 中说明需要用户补充。
"""

SUMMARY_PROMPT = """请根据客服 ReAct Agent 的执行结果，生成简洁、专业的客服回复。
要求：
1. 明确说明已执行哪些动作。
2. 如果有查询结果，概括关键状态。
3. 如果创建了工单，提示用户保存工单号。
4. 如果信息不足，明确说明需要补充什么。
"""


def react_customer_service_node(state: ChatState) -> dict:
    query = state["raw_query"]
    user_id = state.get("user_id") or "anonymous"
    decision = _analyze_actions(query, state.get("entities") or {})
    observations = _execute_actions(decision, query, user_id, state.get("turn_id", "untracked"))
    answer = _summarize(query, decision, observations)
    return {
        "sub_results": {**state.get("sub_results", {}), "react_customer_service": answer},
        "current_agent": "react_customer_service",
        "node_logs": ["ReAct客服处理完成"],
    }


def _analyze_actions(query: str, entities: dict[str, str]) -> dict[str, Any]:
    response = invoke_llm([
        SystemMessage(content=REACT_SYSTEM_PROMPT),
        HumanMessage(content=f"用户消息: {query}\n已识别实体: {json.dumps(entities, ensure_ascii=False)}"),
    ], run_name="react.decide")
    return _parse_json(str(response.content), query)


class ReactDecision(BaseModel):
    steps: list[str] = Field(default_factory=list, max_length=20)
    knowledge_query: str | None = Field(default=None, max_length=1000)
    need_order_query: StrictBool = False
    order_id: str | None = Field(default=None, max_length=128)
    need_ticket_query: StrictBool = False
    ticket_id: str | None = Field(default=None, max_length=64)
    need_ticket_create: StrictBool = False
    ticket_type: Literal["refund", "claim", "account_open", "account_change", "complaint", "general"] = "general"
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    summary: str | None = Field(default=None, max_length=255)
    details: str | None = Field(default=None, max_length=65_535)
    need_risk_check: StrictBool = False
    risk_action: str | None = Field(default=None, max_length=128)
    amount: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    final_answer: str = Field(default="", max_length=4000)


def _parse_json(content: str, query: str) -> dict[str, Any]:
    parsed = parse_json_object(content)
    if parsed is not None:
        try:
            result = ReactDecision.model_validate(parsed).model_dump()
            record_json_parse("react.decide", True)
            return result
        except ValidationError:
            pass
    record_json_parse("react.decide", False)
    return {"steps": ["知识库检索"], "knowledge_query": query, "final_answer": "我将先查询相关资料后为您处理。"}


def _execute_actions(decision: dict[str, Any], query: str, user_id: str, turn_id: str = "untracked") -> dict[str, Any]:
    observations: dict[str, Any] = {}
    knowledge_query = decision.get("knowledge_query")
    if knowledge_query:
        observations["knowledge"] = knowledge_service.get_service().search(str(knowledge_query), top_k=3)

    if decision.get("need_order_query"):
        order_id = decision.get("order_id") or _extract_order_id(query)
        observations["order"] = order_service.get_service().query_order(order_id=order_id or "", user_id=user_id)

    if decision.get("need_ticket_query"):
        ticket_id = decision.get("ticket_id") or _extract_ticket_id(query)
        observations["ticket"] = ticket_service.get_service().query(ticket_id) if ticket_id else None
        observations["ticket_id"] = ticket_id

    if decision.get("need_risk_check"):
        observations["risk"] = risk_service.get_service().check(
            user_id=user_id,
            action=str(decision.get("risk_action") or "customer_service_action"),
            amount=float(decision.get("amount") or 0),
        )

    if decision.get("need_ticket_create"):
        observations["created_ticket"] = ticket_service.get_service().create(
            ticket_type=str(decision.get("ticket_type") or "general"),
            priority=str(decision.get("priority") or "medium"),
            summary=str(decision.get("summary") or query[:100]),
            details=str(decision.get("details") or query),
            user_id=user_id,
            operation_key=f"{turn_id}:react_customer_service:create:0",
        )
    return observations


def _summarize(query: str, decision: dict[str, Any], observations: dict[str, Any]) -> str:
    response = invoke_llm([
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=(
            f"用户问题: {query}\n\n"
            f"动作决策: {json.dumps(decision, ensure_ascii=False)}\n\n"
            f"执行结果: {json.dumps(observations, ensure_ascii=False, default=str)}"
        )),
    ], run_name="react.summarize")
    return str(response.content)


def _extract_order_id(query: str) -> str:
    match = re.search(r"ORD[-_A-Za-z0-9]+", query, re.IGNORECASE)
    return match.group(0) if match else ""


def _extract_ticket_id(query: str) -> str:
    match = re.search(r"TK[-_A-Za-z0-9]+", query, re.IGNORECASE)
    return match.group(0) if match else ""
