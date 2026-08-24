from __future__ import annotations

from datetime import datetime, timezone

from application.customer_service.agent_run_service import AgentRunRegistry
from domain.customer_service_agent.interfaces.i_conversation_archive import IConversationArchive
from domain.customer_service_agent.service import conversation_archive_service
from infra.memory.persistent_conversation_memory import PersistentConversationMemory
from infra.memory.short_term_memory import RedisShortTermMemory


class FakeArchive(IConversationArchive):
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, object]] = {}
        self.messages: dict[str, list[dict[str, str]]] = {}
        self.runs: dict[str, dict[str, object]] = {}
        self.run_steps: dict[str, list[dict[str, object]]] = {}

    @property
    def available(self) -> bool:
        return True

    def create_session(self, session_id: str, title: str, agent_id: str) -> dict[str, object]:
        now = datetime.now(timezone.utc).isoformat()
        return self.sessions.setdefault(session_id, {
            "id": session_id,
            "title": title,
            "agent_id": agent_id,
            "created_at": now,
            "updated_at": now,
            "favorite": False,
            "message_count": 0,
        })

    def get_session(self, session_id: str) -> dict[str, object] | None:
        return self.sessions.get(session_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: str,
        turn_id: str | None = None,
    ) -> bool:
        session = self.create_session(session_id, "新会话", "general")
        messages = self.messages.setdefault(session_id, [])
        if turn_id and any(message.get("turn_id") == turn_id and message["role"] == role for message in messages):
            return True
        message = {"session_id": session_id, "role": role, "content": content, "timestamp": timestamp}
        if turn_id:
            message["turn_id"] = turn_id
        messages.append(message)
        if role == "user" and session["message_count"] == 0:
            session["title"] = content
        session["message_count"] = int(session["message_count"]) + 1
        session["updated_at"] = timestamp
        return True

    def get_message_by_turn(self, turn_id: str, role: str) -> dict[str, str] | None:
        for messages in self.messages.values():
            for message in messages:
                if message.get("turn_id") == turn_id and message["role"] == role:
                    return dict(message)
        return None

    def get_history(self, session_id: str, last_n: int) -> list[dict[str, str]]:
        return [dict(message) for message in self.messages.get(session_id, [])[-last_n:]]

    def list_sessions(self) -> list[dict[str, object]]:
        return list(self.sessions.values())

    def update_session(self, session_id: str, *, title=None, favorite=None):
        session = self.sessions.get(session_id)
        if session is None:
            return None
        if title is not None:
            session["title"] = title
        if favorite is not None:
            session["favorite"] = favorite
        return session

    def delete_session(self, session_id: str) -> bool:
        self.messages.pop(session_id, None)
        return self.sessions.pop(session_id, None) is not None

    def start_run(self, session_id: str, turn_id: str, started_at: str) -> None:
        self.runs[turn_id] = {"session_id": session_id, "status": "running", "started_at": started_at}

    def mark_run_stop_requested(self, turn_id: str) -> None:
        self.runs[turn_id]["stop_requested"] = True

    def finish_run(self, turn_id: str, status: str, completed_at: str) -> None:
        self.runs[turn_id].update(status=status, completed_at=completed_at)

    def record_run_step(self, turn_id, sequence_no, step_type, status, title, payload, created_at) -> None:
        steps = self.run_steps.setdefault(turn_id, [])
        mapped = {
            "id": f"{turn_id}-{sequence_no}",
            "type": step_type,
            "status": status,
            "title": title,
            "toolName": payload.get("node_name"),
            "startedAt": created_at,
            "completedAt": payload.get("completed_at"),
            "duration": payload.get("duration_ms"),
            "tokenUsage": payload.get("token_usage"),
            "modelCalls": payload.get("model_calls", 0),
        }
        if sequence_no < len(steps):
            steps[sequence_no] = mapped
        else:
            steps.append(mapped)

    def list_runs(self, session_id: str, limit: int = 50):
        return [
            {"turn_id": turn_id, "session_id": session_id, **run, "steps": self.run_steps.get(turn_id, [])}
            for turn_id, run in self.runs.items()
            if run["session_id"] == session_id
        ][:limit]


def test_message_write_through_to_archive_and_cache() -> None:
    cache = RedisShortTermMemory(None)
    archive = FakeArchive()
    memory = PersistentConversationMemory(cache, archive)

    memory.add_message("session-1", "user", "我要开户")

    assert archive.get_history("session-1", 20)[0]["content"] == "我要开户"
    assert cache.get_history("session-1")[0]["content"] == "我要开户"


def test_cache_miss_loads_mysql_archive_and_refills_redis_cache() -> None:
    cache = RedisShortTermMemory(None)
    archive = FakeArchive()
    archive.add_message(
        "session-2",
        "user",
        "永久保存的消息",
        "2026-08-09T15:26:07.910474+00:00",
    )
    memory = PersistentConversationMemory(cache, archive)

    history = memory.get_history("session-2")

    assert history[0]["content"] == "永久保存的消息"
    assert cache.get_history("session-2")[0]["timestamp"] == "2026-08-09T15:26:07.910474+00:00"


def test_long_history_query_uses_archive_beyond_cache_window() -> None:
    cache = RedisShortTermMemory(None, max_turns=2)
    archive = FakeArchive()
    for index in range(4):
        archive.add_message("session-3", "user", f"message-{index}", datetime.now(timezone.utc).isoformat())
    cache.restore_history("session-3", archive.get_history("session-3", 2))
    memory = PersistentConversationMemory(cache, archive)

    history = memory.get_history("session-3", last_n=4)

    assert [message["content"] for message in history] == ["message-0", "message-1", "message-2", "message-3"]


def test_summary_cache_is_user_scoped_and_removed_with_session() -> None:
    cache = RedisShortTermMemory(None, user_id="user-1")
    summary = {
        "user_id": "user-1",
        "session_id": "session-summary",
        "version": 1,
        "summary_text": "滚动摘要",
    }

    cache.cache_session_summary("session-summary", summary)

    assert cache.get_session_summary("session-summary") == summary
    cache.delete_session("session-summary")
    assert cache.get_session_summary("session-summary") is None


def test_agent_run_lifecycle_is_archived(monkeypatch) -> None:
    archive = FakeArchive()
    monkeypatch.setattr(conversation_archive_service, "instance", archive)
    registry = AgentRunRegistry()

    run = registry.begin("session-4", "turn-4")
    step = registry.start_step(run, "监督路由", node_name="supervisor_node", step_type="plan")
    registry.complete_step(
        run,
        step["id"],
        duration_ms=125,
        token_usage={"prompt": 10, "completion": 5, "total": 15},
        model_calls=1,
    )
    registry.stop("session-4")
    registry.finish(run, "stopped")

    assert archive.runs["turn-4"]["status"] == "stopped"
    assert archive.runs["turn-4"]["stop_requested"] is True
    assert archive.runs["turn-4"]["completed_at"]
    assert archive.run_steps["turn-4"][0]["duration"] == 125
    assert archive.run_steps["turn-4"][0]["tokenUsage"]["total"] == 15
