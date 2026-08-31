from __future__ import annotations

import uuid

from threading import Lock
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, Request, Response, status

from adapter.web.routes.response_helpers import agent_idle_response, session_response
from adapter.web.schemas.workbench import CurrentUserUpdateRequest, SessionCreateRequest, SessionUpdateRequest, StopAgentRequest
from application.customer_service import agent_run_service, workbench_service
from application.customer_service.turn_lease_service import get_turn_lease_manager
from domain.customer_service_agent.service import conversation_archive_service, short_term_memory_service
from domain.shared.checkpoint import checkpoint_saver_service
from pkg.security.request_identity import resolve_request_user

router = APIRouter(prefix="/api")
_session_creation_lock = Lock()
SessionId = Annotated[str, Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")]


@router.get("/agents")
def list_agents():
    return {"agents": workbench_service.list_agents()}


@router.get("/files")
def list_files():
    return {"files": workbench_service.list_workspace_files()}


@router.get("/settings/current-user")
def get_current_user():
    return workbench_service.get_current_user()


@router.put("/settings/current-user")
def update_current_user(request: CurrentUserUpdateRequest):
    return workbench_service.update_current_user(request.user_id)


@router.get("/sessions")
def list_sessions(
    http_request: Request,
    user_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$"),
):
    resolved_user_id, _, _ = resolve_request_user(http_request, user_id)
    sessions = short_term_memory_service.get_service().list_sessions(user_id=resolved_user_id)
    return {"sessions": [session_response(session) for session in sessions]}


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
def create_session(request: SessionCreateRequest, http_request: Request):
    user_id, _, _ = resolve_request_user(http_request, request.user_id)
    memory = short_term_memory_service.get_service()
    with _session_creation_lock:
        session = memory.create_session(
            request.session_id,
            title=request.title,
            agent_id=request.agent_id,
            user_id=user_id,
        )
    return session_response(session)


@router.patch("/sessions/{session_id}")
def update_session(session_id: SessionId, request: SessionUpdateRequest, http_request: Request):
    user_id, _, _ = resolve_request_user(http_request, request.user_id)
    session = short_term_memory_service.get_service().update_session(
        session_id,
        title=request.title,
        favorite=request.favorite,
        user_id=user_id,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session_response(session)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: SessionId,
    http_request: Request,
    user_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$"),
) -> Response:
    resolved_user_id, _, _ = resolve_request_user(http_request, user_id)
    run = agent_run_service.registry.get(session_id, user_id=resolved_user_id)
    if run is not None and run.status == "running":
        raise HTTPException(status_code=409, detail="请先停止正在运行的 Agent")
    manager = get_turn_lease_manager()
    lease = None
    if manager.available:
        try:
            lease = manager.acquire(
                resolved_user_id,
                session_id,
                f"delete-{uuid.uuid4().hex}",
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail="请先停止正在运行的 Agent") from exc
    try:
        removed = short_term_memory_service.get_service().delete_session(session_id, user_id=resolved_user_id)
    finally:
        if lease is not None:
            manager.release(lease)
    if not removed:
        raise HTTPException(status_code=404, detail="会话不存在")
    agent_run_service.registry.forget(session_id, user_id=resolved_user_id)
    if checkpoint_saver_service.instance is not None:
        checkpoint_saver_service.instance.clear_thread(resolved_user_id, session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/agent/stop")
def stop_agent(request: StopAgentRequest, http_request: Request):
    user_id, _, _ = resolve_request_user(http_request, request.user_id)
    run = agent_run_service.registry.stop(request.session_id, user_id=user_id)
    durable_stopped = get_turn_lease_manager().request_stop(
        user_id,
        request.session_id,
    )
    if run is None and not durable_stopped:
        return agent_idle_response(request.session_id)
    if run is not None:
        return run.as_dict()
    durable_run = get_turn_lease_manager().status(user_id, request.session_id)
    return durable_run or agent_idle_response(request.session_id)


@router.get("/agent/status/{session_id}")
def get_agent_status(
    session_id: SessionId,
    http_request: Request,
    user_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$"),
):
    resolved_user_id, _, _ = resolve_request_user(http_request, user_id)
    run = agent_run_service.registry.get(session_id, user_id=resolved_user_id)
    if run is not None:
        return run.as_dict()
    durable_run = get_turn_lease_manager().status(resolved_user_id, session_id)
    return durable_run or agent_idle_response(session_id)


@router.get("/runs")
def list_agent_runs(
    http_request: Request,
    session_id: str = Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
    limit: int = Query(20, ge=1, le=100),
    user_id: str | None = Query(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$"),
):
    resolved_user_id, _, _ = resolve_request_user(http_request, user_id)
    archive = conversation_archive_service.get_service_or_none()
    runs = (
        archive.list_runs(session_id, limit=limit, user_id=resolved_user_id)
        if archive is not None and archive.available
        else []
    )
    if not runs:
        runs = agent_run_service.registry.list_runs(session_id, user_id=resolved_user_id)[:limit]
    return {"runs": runs}
