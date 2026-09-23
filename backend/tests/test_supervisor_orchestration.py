from __future__ import annotations

import asyncio
from types import SimpleNamespace

from domain.customer_service_agent.orchestration import pending_action_resolver
from domain.customer_service_agent.workflow.entity.chat_state import create_chat_state
from domain.shared.identity import RequestIdentityContext


def _identity() -> RequestIdentityContext:
    return RequestIdentityContext(user_id="user-1", session_id="session-1", turn_id="turn-1")


def test_pending_action_resolver_degrades_when_service_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        pending_action_resolver,
        "get_action_service",
        lambda: (_ for _ in ()).throw(RuntimeError("not initialized")),
    )

    result = asyncio.run(
        pending_action_resolver.resolve_pending_action(
            create_chat_state("user-1", "session-1", "你好"),
            _identity(),
        )
    )

    assert result is None


def test_pending_action_is_resolved_before_normal_routing(monkeypatch) -> None:
    active = SimpleNamespace(impact_summary="变更套餐")

    class Actions:
        def get_active(self, _identity):
            return active

    monkeypatch.setattr(pending_action_resolver, "get_action_service", lambda: Actions())

    result = asyncio.run(
        pending_action_resolver.resolve_pending_action(
            create_chat_state("user-1", "session-1", "再帮我查订单"),
            _identity(),
        )
    )

    assert result["intent"] == "action_pending"
    assert "请仅回复“确认”" in result["sub_results"]["supervisor"]
