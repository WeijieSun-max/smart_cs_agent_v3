from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, Request

from adapter.web.routes.response_helpers import raise_running_agent_conflict
from adapter.web.schemas.chat import ChatRequest, ChatStreamRequest, FeedbackRequest, FeedbackResponse, ToolCallRequest
from application.customer_service import chat_service, memory_admin_service
from domain.customer_service_agent.service import short_term_memory_service
from domain.customer_service_agent.tools.tool_registry import get_mcp_server
from pkg.telemetry import memory_metrics_snapshot, submit_user_feedback
from pkg.security.request_identity import resolve_request_user
from pkg.security import get_local_user_id
from application.customer_service.session_ownership import get_session_ownership

router = APIRouter()
SessionId = Annotated[str, Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")]


@router.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request):
    try:
        user_id, _, _ = resolve_request_user(http_request, request.user_id)
        normalized_request = request.model_copy(update={"user_id": user_id})
        return await chat_service.chat(normalized_request)
    except RuntimeError as exc:
        raise_running_agent_conflict(exc)


@router.post("/api/chat_stream")
def chat_stream(request: ChatStreamRequest, http_request: Request):
    try:
        user_id, _, _ = resolve_request_user(http_request, request.user_id)
        normalized_request = request.model_copy(update={"user_id": user_id})
        return chat_service.chat_stream(normalized_request)
    except RuntimeError as exc:
        raise_running_agent_conflict(exc)


@router.get("/api/history/{session_id}")
def get_history(
    session_id: SessionId,
    user_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$"),
    limit: int = Query(default=100, ge=1, le=500),
):
    get_session_ownership().require_owner(user_id or get_local_user_id(), session_id)
    messages = short_term_memory_service.get_service().get_history(session_id, last_n=limit)
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": f"{session_id}-{index}",
                "role": message["role"],
                "content": message["content"],
                "created_at": message.get("timestamp"),
            }
            for index, message in enumerate(messages)
        ],
    }


@router.get("/api/tools")
def list_tools():
    return {"tools": get_mcp_server().list_tools()}


@router.post("/api/tools/call")
async def call_tool(request: ToolCallRequest):
    del request
    raise HTTPException(status_code=404, detail="Direct tool execution is disabled")


@router.get("/api/metrics")
def get_metrics():
    return {
        "tool_call_log": get_mcp_server().get_call_log(last_n=20),
        "memory": memory_metrics_snapshot(),
        "memory_outbox": memory_admin_service.current_outbox_status(),
    }


@router.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    accepted = submit_user_feedback(request.trace_id, request.helpful, request.reason)
    if not accepted:
        raise HTTPException(status_code=503, detail="Feedback telemetry is unavailable")
    return FeedbackResponse(accepted=True)
