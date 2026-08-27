from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from adapter.web.schemas.chat import ChatRequest
from application.customer_service import chat_service
from domain.customer_service_agent.memory.models import (
    MemoryItem,
    MemoryStatus,
    MemoryType,
    SessionSummary,
    VectorSearchHit,
)
from domain.customer_service_agent.service import memory_service, short_term_memory_service
from domain.customer_service_agent.service.memory_orchestrator import MemoryOrchestrator
from infra.memory.short_term_memory import RedisShortTermMemory
from pkg.security import get_local_user_id
from scripts.evaluate_memory_retrieval import evaluate


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


def _settings(**overrides):
    values = {
        "memory_layered_enabled": True,
        "memory_recent_messages_tokens": 700,
        "memory_session_summary_tokens": 350,
        "memory_episode_tokens": 350,
        "memory_semantic_tokens": 400,
        "memory_context_max_tokens": 1800,
        "memory_recall_top_k": 20,
        "memory_recall_min_score": 0.35,
        "memory_fallback_items": 10,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _item(
    memory_id: str,
    user_id: str,
    memory_type: MemoryType,
    content: str,
    *,
    updated_at: datetime = NOW,
    expires_at: datetime | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        user_id=user_id,
        memory_type=memory_type,
        memory_key=f"e2e.{memory_id}",
        content=content,
        confidence=0.95,
        status=MemoryStatus.ACTIVE,
        version=1,
        expires_at=expires_at or NOW + timedelta(days=30),
        created_at=updated_at,
        updated_at=updated_at,
    )


class Repository:
    available = True

    def __init__(self, items, summary=None):
        self.items = items
        self.summary = summary

    def get_latest_summary(self, user_id, session_id):
        if self.summary and self.summary.user_id == user_id and self.summary.session_id == session_id:
            return self.summary
        return None

    def get_items_by_ids(self, user_id, memory_ids, now):
        del user_id, now
        wanted = set(memory_ids)
        return [item for item in self.items if str(item.memory_id) in wanted]

    def list_active_items(self, user_id, *, memory_types=None, limit=20, now):
        del memory_types, now
        return [item for item in self.items if item.user_id == user_id][:limit]


class Index:
    available = True

    def __init__(self, hits):
        self.hits = hits

    def search(self, **kwargs):
        return self.hits


class Embedder:
    def embed_query(self, text):
        return [1.0, 0.0]


class Recent:
    def __init__(self, history):
        self.history = history

    def get_history(self, session_id):
        return self.history


def test_cross_session_recall_is_isolated_relevant_and_budgeted() -> None:
    valid = _item(
        "60000000-0000-0000-0000-000000000001",
        "user-a",
        MemoryType.PREFERENCE,
        "用户偏好短信接收退款进度",
    )
    irrelevant_recent = _item(
        "60000000-0000-0000-0000-000000000002",
        "user-a",
        MemoryType.FACT,
        "最近浏览过开户页面",
        updated_at=NOW + timedelta(seconds=1),
    )
    other_user = _item(
        "60000000-0000-0000-0000-000000000003",
        "user-b",
        MemoryType.TASK,
        "另一个用户的退款任务",
    )
    expired = _item(
        "60000000-0000-0000-0000-000000000004",
        "user-a",
        MemoryType.FACT,
        "已过期退款事实",
        expires_at=NOW - timedelta(seconds=1),
    )
    summary = SessionSummary(
        user_id="user-a",
        session_id="new-session",
        version=1,
        summary_text="用户正在继续处理退款问题。" * 200,
        covers_until_message_id=10,
        created_at=NOW,
        updated_at=NOW,
    )
    hits = [
        VectorSearchHit(memory_id=valid.memory_id, score=0.95),
        VectorSearchHit(memory_id=irrelevant_recent.memory_id, score=0.20),
        VectorSearchHit(memory_id=other_user.memory_id, score=0.99),
        VectorSearchHit(memory_id=expired.memory_id, score=0.99),
    ]
    orchestrator = MemoryOrchestrator(
        repository=Repository([valid, irrelevant_recent, other_user, expired], summary),
        vector_index=Index(hits),
        embedder=Embedder(),
        short_term_memory=Recent([
            {"role": "user", "content": "旧消息" * 1000, "turn_id": "old-turn"},
            {"role": "user", "content": "当前退款问题", "turn_id": "current-turn"},
        ]),
        settings=_settings(),
        clock=lambda: NOW,
    )

    packet = orchestrator.build_packet("user-a", "new-session", "退款进度", "current-turn")

    assert packet.token_count <= 1800
    assert [item.content for item in packet.semantic_memories] == ["用户偏好短信接收退款进度"]
    assert all("当前退款问题" not in message.content for message in packet.recent_messages)
    assert len(packet.session_summary) < len(summary.summary_text)


def test_prepare_turn_uses_layered_packet_when_feature_is_enabled(monkeypatch) -> None:
    user_id = get_local_user_id()
    item = _item(
        "70000000-0000-0000-0000-000000000001",
        user_id,
        MemoryType.TASK,
        "继续跟进退款进度",
    )
    repository = Repository([item])
    memory_service.initialize_service(repository, vector_index=None, embedder=None)
    short_memory = RedisShortTermMemory(None)
    short_memory.add_message("new-session", "user", "此前咨询退款")
    short_term_memory_service.initialize_service(short_memory)
    monkeypatch.setattr(chat_service, "get_settings", lambda: _settings())

    _, state, _, replay = chat_service._prepare_turn(
        ChatRequest(message="查询退款进度", session_id="new-session")
    )

    assert replay is None
    assert state["conversation_context"] == {
        "summary": "",
        "recent_messages": [],
        "memories": [],
    }
    assert state["memory_packet"] is not None
    assert state["memory_packet"]["semantic_memories"][0]["content"] == "继续跟进退款进度"


def test_fixed_retrieval_fixture_meets_recall_and_latency_targets() -> None:
    fixture = Path(__file__).parent / "fixtures" / "memory_retrieval_cases.json"

    result = evaluate(fixture)

    assert result["recall_at_5"] >= 0.80
    assert result["p95_ms"] <= 800
