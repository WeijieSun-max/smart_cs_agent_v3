from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from adapter.web.schemas.chat import ChatRequest, ChatStreamRequest
from application.customer_service import chat_service


class AdmissionProbe:
    def __init__(self) -> None:
        self.active = False
        self.released = False

    @asynccontextmanager
    async def slot(self, _user_id: str):
        assert self.active is False
        self.active = True
        try:
            yield
        finally:
            self.active = False
            self.released = True


def test_chat_holds_admission_during_complete_application_call(monkeypatch) -> None:
    probe = AdmissionProbe()

    async def admitted(_request):
        assert probe.active is True
        return "response"

    monkeypatch.setattr(chat_service, "turn_admission", probe)
    monkeypatch.setattr(chat_service, "_chat_admitted", admitted)

    result = asyncio.run(chat_service.chat(ChatRequest(message="查询套餐", user_id="user-1")))

    assert result == "response"
    assert probe.released is True


def test_stream_prepares_and_replays_inside_admission(monkeypatch) -> None:
    probe = AdmissionProbe()
    turn = SimpleNamespace(turn_id="turn-1", trace_id="trace-1")

    def prepare(_request):
        assert probe.active is True
        return "user-1", {"session_id": "session-1"}, turn, {"content": "cached"}

    monkeypatch.setattr(chat_service, "turn_admission", probe)
    monkeypatch.setattr(chat_service, "_prepare_turn", prepare)
    monkeypatch.setattr(chat_service.customer_service_workflow, "get_workflow", lambda: object())

    async def collect() -> str:
        return "".join([
            item
            async for item in chat_service._generate_stream_events(
                ChatStreamRequest(message="查询套餐", user_id="user-1")
            )
        ])

    output = asyncio.run(collect())

    assert '"replayed": true' in output
    assert '"content": "cached"' in output
    assert probe.released is True


def test_stream_accepts_business_contact_details_before_streaming() -> None:
    response = chat_service.chat_stream(
        ChatStreamRequest(
            message="张伟 18060815554 文艺路9号南京邮电大学仙林校区东门",
            user_id="user-1",
        )
    )

    assert response.media_type == "text/event-stream"


def test_stream_releases_admission_when_prepare_fails(monkeypatch) -> None:
    probe = AdmissionProbe()

    def prepare(_request):
        assert probe.active is True
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(chat_service, "turn_admission", probe)
    monkeypatch.setattr(chat_service, "_prepare_turn", prepare)
    monkeypatch.setattr(chat_service.customer_service_workflow, "get_workflow", lambda: object())

    async def collect() -> None:
        async for _ in chat_service._generate_stream_events(
            ChatStreamRequest(message="查询套餐", user_id="user-1")
        ):
            pass

    with pytest.raises(RuntimeError, match="prepare failed"):
        asyncio.run(collect())
    assert probe.released is True
