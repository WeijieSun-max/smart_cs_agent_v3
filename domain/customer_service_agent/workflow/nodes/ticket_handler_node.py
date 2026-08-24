from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from domain.customer_service_agent.service import ticket_service
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.llm.llm_service import invoke_llm
from pkg.telemetry import record_json_parse
from pkg.exceptions.exception import StructuredOutputError
from pkg.llm import parse_json_object

TICKET_SYSTEM_PROMPT = """你是一个专业的工单处理Agent，负责处理客户的业务办理请求。

工单类型：refund, claim, account_open, account_change, complaint, general
优先级：low, medium, high, urgent

请只返回JSON：
{
  "action": "create|query|update",
  "ticket_id": "可选",
  "ticket_type": "refund|claim|account_open|...",
  "priority": "low|medium|high|urgent",
  "summary": "工单摘要",
  "details": "详细描述",
  "status": "可选"
}
"""


def ticket_handler_node(state: ChatState) -> dict:
    query = state["raw_query"]
    info = _analyze_ticket(query)
    repo = ticket_service.get_service()
    action = info.get("action", "create")

    if action == "query":
        ticket_id = info.get("ticket_id")
        result = _format_ticket_query(repo.query(ticket_id), ticket_id) if ticket_id else "请提供需要查询的工单号。"
    elif action == "update":
        ticket_id = info.get("ticket_id")
        status = info.get("status")
        result = (
            _format_ticket_query(repo.update_status(ticket_id, status), ticket_id)
            if ticket_id and status
            else "请提供工单号和需要更新的目标状态。"
        )
    else:
        ticket = repo.create(
            ticket_type=info.get("ticket_type", "general"),
            priority=info.get("priority", "medium"),
            summary=info.get("summary") or query[:100],
            details=info.get("details") or query,
            user_id=state.get("user_id") or "anonymous",
            operation_key=f"{state.get('turn_id', 'untracked')}:ticket_handler:create:0",
        )
        result = _format_ticket_created(ticket)

    return {
        "sub_results": {**state.get("sub_results", {}), "ticket_handler": result},
        "current_agent": "ticket_handler",
        "node_logs": ["工单处理完成"],
    }


class TicketDecision(BaseModel):
    action: Literal["create", "query", "update"]
    ticket_id: str | None = Field(default=None, max_length=64)
    ticket_type: Literal["refund", "claim", "account_open", "account_change", "complaint", "general"] = "general"
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    summary: str = Field(default="", max_length=255)
    details: str = Field(default="", max_length=65_535)
    status: Literal["created", "processing", "pending_review", "resolved", "closed", "escalated"] | None = None


def _analyze_ticket(query: str) -> dict:
    response = invoke_llm([
        SystemMessage(content=TICKET_SYSTEM_PROMPT),
        HumanMessage(content=f"用户消息: {query}"),
    ], run_name="ticket.analyze")
    content = str(response.content)
    parsed = parse_json_object(content)
    if parsed is not None:
        try:
            result = TicketDecision.model_validate(parsed).model_dump()
        except ValidationError as exc:
            record_json_parse("ticket.analyze", False)
            raise StructuredOutputError() from exc
        record_json_parse("ticket.analyze", True)
        return result
    record_json_parse("ticket.analyze", False)
    raise StructuredOutputError()


def _format_ticket_created(ticket: dict) -> str:
    priority_label = {"low": "普通", "medium": "中等", "high": "高", "urgent": "紧急"}.get(ticket["priority"], "中等")
    return (
        "工单已创建成功！\n\n"
        f"工单号: {ticket['ticket_id']}\n"
        f"类型: {ticket['type']}\n"
        f"优先级: {priority_label}\n"
        f"摘要: {ticket['summary']}\n"
        f"创建时间: {ticket['created_at']}\n\n"
        "我们将尽快处理您的请求，请保存好工单号以便后续查询。"
    )


def _format_ticket_query(ticket: dict | None, ticket_id: str) -> str:
    if not ticket:
        return f"未找到工单号 {ticket_id}，请确认工单号是否正确。"
    status_label = {
        "created": "已创建",
        "processing": "处理中",
        "pending_review": "待审核",
        "resolved": "已解决",
        "closed": "已关闭",
        "escalated": "已升级",
    }.get(ticket["status"], ticket["status"])
    return (
        "工单查询结果：\n\n"
        f"工单号: {ticket['ticket_id']}\n"
        f"状态: {status_label}\n"
        f"类型: {ticket['type']}\n"
        f"摘要: {ticket['summary']}\n"
        f"创建时间: {ticket['created_at']}\n"
        f"更新时间: {ticket['updated_at']}"
    )
