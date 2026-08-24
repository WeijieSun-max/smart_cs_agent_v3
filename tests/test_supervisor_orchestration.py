from __future__ import annotations

import asyncio
from types import SimpleNamespace

from domain.customer_service_agent.orchestration import pending_action_resolver, skill_executor, write_proposal
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


def test_plan_skill_unavailable_returns_stable_result(monkeypatch) -> None:
    catalog = SimpleNamespace(select=lambda **_kwargs: None)
    monkeypatch.setattr(skill_executor, "get_catalog", lambda: catalog)

    result = asyncio.run(
        skill_executor.execute_plan_recommendation(
            create_chat_state("user-1", "session-1", "推荐套餐"),
            _identity(),
        )
    )

    assert result["intent"] == "telecom"
    assert result["sub_results"]["supervisor"] == "套餐推荐能力暂不可用。"


def test_write_proposal_validates_required_entities_before_service_lookup(monkeypatch) -> None:
    monkeypatch.setattr(
        write_proposal,
        "get_action_service",
        lambda: (_ for _ in ()).throw(AssertionError("service must not be read")),
    )

    result = asyncio.run(
        write_proposal.prepare_write_proposal("帮我换套餐", "plan_change", _identity())
    )

    assert "line_id 和 plan_id" in result


def test_workspace_only_write_does_not_touch_action_service(monkeypatch) -> None:
    monkeypatch.setattr(
        write_proposal,
        "get_action_service",
        lambda: (_ for _ in ()).throw(AssertionError("service must not be read")),
    )

    result = asyncio.run(
        write_proposal.prepare_write_proposal(
            "修改订单地址",
            "update_order_address",
            _identity(),
        )
    )

    assert "请在工作台填写" in result
