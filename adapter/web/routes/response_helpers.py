from __future__ import annotations

from typing import Any, NoReturn

from fastapi import HTTPException


def raise_running_agent_conflict(exc: RuntimeError) -> NoReturn:
    if "already has a running agent" in str(exc):
        raise HTTPException(status_code=409, detail="当前会话已有 Agent 正在运行") from exc
    raise exc


def agent_idle_response(session_id: str) -> dict[str, str | bool]:
    return {
        "session_id": session_id,
        "status": "idle",
        "running": False,
        "stop_requested": False,
    }


def session_response(session: dict[str, object]) -> dict[str, object]:
    return {
        "id": session["id"],
        "title": session["title"],
        "agentId": session["agent_id"],
        "createdAt": session["created_at"],
        "updatedAt": session["updated_at"],
        "favorite": session["favorite"],
        "messageCount": session["message_count"],
    }


def public_action_response(action: Any) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "status": action.status,
        "tool_name": action.tool_name,
        "impact_summary": action.impact_summary,
        "expires_at": action.expires_at,
        "receipt": action.receipt,
        "error_code": action.error_code,
    }
