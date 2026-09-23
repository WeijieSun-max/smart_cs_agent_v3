from __future__ import annotations

import asyncio
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from adapter.web.routes import chat_controller, workbench_controller
from application.customer_service import agent_run_service, chat_service, workbench_service
from domain.customer_service_agent.service import short_term_memory_service
from infra.memory.short_term_memory import RedisShortTermMemory
from pkg.security import get_local_user_id
from pkg.telemetry import create_turn_trace


def _client() -> TestClient:
    short_term_memory_service.initialize_service(RedisShortTermMemory(None))
    app = FastAPI()
    app.include_router(chat_controller.router)
    app.include_router(workbench_controller.router)
    return TestClient(app)


def test_chinese_message_serialization_stays_utf8_text() -> None:
    memory = RedisShortTermMemory(None)
    message = {
        "session_id": "serialization-test",
        "role": "user",
        "content": "我要开户，需要准备什么",
        "timestamp": "2026-08-09T15:26:07.910474+00:00",
    }

    serialized = memory._serialize_message(message)
    assert "我要开户，需要准备什么" in serialized
    assert "\\u6211" not in serialized
    assert memory._deserialize_message(serialized.encode("utf-8")) == message


def test_session_crud_and_history_contract() -> None:
    client = _client()
    session_id = uuid.uuid4().hex

    created = client.post("/api/sessions", json={"session_id": session_id, "agent_id": "general"})
    assert created.status_code == 201
    assert created.json()["id"] == session_id
    assert created.json()["messageCount"] == 0

    memory = short_term_memory_service.get_service()
    memory.add_message(session_id, "user", "请帮我查询订单状态")
    memory.add_message(session_id, "assistant", "请提供订单号。")

    sessions = client.get("/api/sessions").json()["sessions"]
    assert sessions[0]["title"] == "请帮我查询订单状态"
    assert sessions[0]["messageCount"] == 2

    history = client.get(f"/api/history/{session_id}").json()["messages"]
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert all(message["created_at"] for message in history)

    updated = client.patch(f"/api/sessions/{session_id}", json={"favorite": True})
    assert updated.status_code == 200
    assert updated.json()["favorite"] is True

    stopped = client.post("/api/agent/stop", json={"session_id": session_id})
    assert stopped.status_code == 200
    assert stopped.json()["session_id"] == session_id

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert client.get("/api/sessions").json()["sessions"] == []


def test_session_crud_supports_explicit_request_user() -> None:
    client = _client()
    session_id = uuid.uuid4().hex

    created = client.post(
        "/api/sessions",
        json={"user_id": "request-user", "session_id": session_id, "agent_id": "general"},
    )
    assert created.status_code == 201

    owner_sessions = client.get("/api/sessions", params={"user_id": "request-user"}).json()["sessions"]
    other_sessions = client.get("/api/sessions", params={"user_id": "other-user"}).json()["sessions"]
    assert [session["id"] for session in owner_sessions] == [session_id]
    assert other_sessions == []

    denied = client.patch(
        f"/api/sessions/{session_id}",
        json={"user_id": "other-user", "favorite": True},
    )
    updated = client.patch(
        f"/api/sessions/{session_id}",
        json={"user_id": "request-user", "favorite": True},
    )
    assert denied.status_code == 404
    assert updated.status_code == 200
    assert updated.json()["favorite"] is True

    assert client.delete(f"/api/sessions/{session_id}", params={"user_id": "request-user"}).status_code == 204


def test_agent_status_and_stop_are_scoped_to_request_user() -> None:
    client = _client()
    session_id = uuid.uuid4().hex
    run = agent_run_service.registry.begin(session_id, uuid.uuid4().hex, user_id="request-user")

    try:
        other_status = client.get(f"/api/agent/status/{session_id}", params={"user_id": "other-user"})
        other_stop = client.post(
            "/api/agent/stop",
            json={"user_id": "other-user", "session_id": session_id},
        )
        owner_status = client.get(f"/api/agent/status/{session_id}", params={"user_id": "request-user"})
        owner_stop = client.post(
            "/api/agent/stop",
            json={"user_id": "request-user", "session_id": session_id},
        )

        assert other_status.json()["running"] is False
        assert other_stop.json()["running"] is False
        assert run.stop_requested.is_set() is True
        assert owner_status.json()["turn_id"] == run.turn_id
        assert owner_stop.json()["turn_id"] == run.turn_id
    finally:
        agent_run_service.registry.finish(run, "stopped")
        agent_run_service.registry.forget(session_id, user_id="request-user")


def test_current_user_setting_is_validated_and_applied(monkeypatch) -> None:
    client = _client()
    original_user_id = get_local_user_id()
    monkeypatch.setattr(workbench_service, "_persist_local_user_id", lambda _user_id: None)

    try:
        assert client.get("/api/settings/current-user").json() == {"user_id": original_user_id}

        updated = client.put("/api/settings/current-user", json={"user_id": "customer_2026"})
        assert updated.status_code == 200
        assert updated.json() == {"user_id": "customer_2026"}
        assert get_local_user_id() == "customer_2026"
        assert not hasattr(short_term_memory_service.get_service().memory, "user_id")

        invalid = client.put("/api/settings/current-user", json={"user_id": "invalid user"})
        assert invalid.status_code == 422
        assert get_local_user_id() == "customer_2026"
    finally:
        workbench_service.update_current_user(original_user_id)


def test_stop_endpoint_cancels_active_stream(monkeypatch) -> None:
    monkeypatch.setattr(chat_service, "_build_graph_config", lambda *_args, **_kwargs: {})

    class SlowGraph:
        async def astream_events(self, *_args, **_kwargs):
            await asyncio.sleep(30)
            yield {}

    async def exercise() -> tuple[str, dict[str, str | bool]]:
        session_id = uuid.uuid4().hex
        turn = create_turn_trace("user-1", session_id)
        run = agent_run_service.registry.begin(session_id, turn.turn_id)

        async def collect() -> str:
            chunks = []
            async for chunk in chat_service._generate_stream_events_admitted(
                SlowGraph(),
                {"raw_query": "slow request"},
                session_id,
                "user-1",
                turn,
                run,
            ):
                chunks.append(chunk)
            return "".join(chunks)

        task = asyncio.create_task(collect())
        await asyncio.sleep(0)
        stopped = agent_run_service.registry.stop(session_id)
        assert stopped is run
        output = await asyncio.wait_for(task, timeout=2)
        return output, run.as_dict()

    output, run = asyncio.run(exercise())
    assert '"type": "stopped"' in output
    assert run["status"] == "stopped"
    assert run["running"] is False
