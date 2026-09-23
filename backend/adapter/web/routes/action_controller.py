from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request

from adapter.web.routes.response_helpers import public_action_response
from adapter.web.schemas.actions import ActionDecisionRequest, ActionProposalRequest
from domain.action_governance import get_action_service
from domain.shared.identity import RequestIdentityContext
from pkg.security.request_identity import resolve_request_user

router = APIRouter()


@router.post("/api/actions/propose")
def propose_action(body: ActionProposalRequest, http_request: Request):
    identity = _identity_context(body, http_request)
    action_service = get_action_service()
    definition = action_service.server.get_tool(body.tool_name)
    if definition is None or definition.effect != "write":
        raise HTTPException(status_code=422, detail="Only governed write tools can be proposed")
    impact = f"{definition.description}；冻结参数：{json.dumps(body.arguments, ensure_ascii=False, sort_keys=True)}"
    action = action_service.propose_write(
        body.tool_name,
        body.arguments,
        identity,
        impact_summary=impact,
    )
    return public_action_response(action)


@router.post("/api/actions/{action_id}/decision")
async def decide_action(action_id: str, body: ActionDecisionRequest, http_request: Request):
    identity = _identity_context(body, http_request)
    action_service = get_action_service()
    active = action_service.get_active(identity)
    if active is None or active.action_id != action_id:
        raise HTTPException(status_code=404, detail="Active action not found")
    if body.decision == "confirm":
        action = await action_service.confirm(identity)
    else:
        action = action_service.reject(identity)
    return public_action_response(action)


def _identity_context(body: ActionProposalRequest | ActionDecisionRequest, http_request: Request) -> RequestIdentityContext:
    user_id, source, strength = resolve_request_user(http_request, body.user_id)
    return RequestIdentityContext(
        user_id=user_id,
        session_id=body.session_id,
        turn_id=body.turn_id,
        identity_source=source,
        auth_strength=strength,
    )
