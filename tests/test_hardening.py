from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from adapter.web.routes import workbench_controller
from adapter.web.schemas.chat import ChatRequest
from application.customer_service import chat_service
from application.customer_service.agent_run_service import AgentRunRegistry
from domain.customer_service_agent.service import conversation_archive_service, short_term_memory_service
from domain.customer_service_agent.tools import mcp_server as mcp_server_module
from domain.customer_service_agent.tools.mcp_server import MCPToolServer
from domain.customer_service_agent.workflow.nodes import compliance_checker_node
from domain.customer_service_agent.workflow.nodes.history_fusion_node import history_fusion_node
from infra.knowledge.local_knowledge_store import LocalKnowledgeStore
from infra.memory.persistent_conversation_memory import PersistentConversationMemory
from infra.memory.mysql_conversation_archive import MySQLConversationArchive
from infra.memory.short_term_memory import RedisShortTermMemory
from pkg.exceptions.exception import (
    RequestConflictError,
    StorageOperationError,
)
from pkg.exceptions.exception import handle_global_exception
from tests.test_persistent_memory import FakeArchive


class UnavailableArchive(FakeArchive):
    @property
    def available(self) -> bool:
        return False


class FailingRunArchive(FakeArchive):
    def start_run(self, session_id: str, turn_id: str, started_at: str) -> None:
        raise StorageOperationError()


def test_mysql_unavailable_session_write_returns_503() -> None:
    cache = RedisShortTermMemory(None)
    short_term_memory_service.initialize_service(
        PersistentConversationMemory(cache, UnavailableArchive())
    )
    app = FastAPI()
    handle_global_exception(app)
    app.include_router(workbench_controller.router)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/api/sessions", json={"session_id": "durable-session", "agent_id": "general"})

    assert response.status_code == 503
    assert cache.list_sessions() == []


def test_agent_run_start_failure_is_not_reported_as_running(monkeypatch) -> None:
    monkeypatch.setattr(conversation_archive_service, "instance", FailingRunArchive())
    registry = AgentRunRegistry()

    with pytest.raises(StorageOperationError):
        registry.begin("session-run", "turn-run")

    assert registry.get("session-run") is None


def test_requested_empty_session_ids_are_not_reused() -> None:
    short_term_memory_service.initialize_service(RedisShortTermMemory(None))
    app = FastAPI()
    app.include_router(workbench_controller.router)
    client = TestClient(app)

    first = client.post("/api/sessions", json={"session_id": "empty-one", "agent_id": "general"})
    second = client.post("/api/sessions", json={"session_id": "empty-two", "agent_id": "general"})

    assert first.json()["id"] == "empty-one"
    assert second.json()["id"] == "empty-two"


def test_frontend_user_is_request_identity_and_retry_context_excludes_current_turn() -> None:
    memory = RedisShortTermMemory(None)
    short_term_memory_service.initialize_service(memory)
    memory.add_message_at("session-1", "user", "same request", "2026-08-09T00:00:00+00:00", turn_id="request-12345678")
    request = ChatRequest(
        message="same request",
        user_id="attacker",
        session_id="session-1",
        request_id="request-12345678",
    )

    user_id, state, _, replay = chat_service._prepare_turn(request)

    assert user_id == "attacker"
    assert state["user_id"] == "attacker"
    assert state["user_message_persisted"] is True
    packet_messages = (state.get("memory_packet") or {}).get("recent_messages", [])
    context_messages = state["conversation_context"]["recent_messages"]
    assert all(
        message.get("content") != "same request"
        for message in [*packet_messages, *context_messages]
    )
    assert replay is None


def test_reusing_request_id_for_different_content_is_rejected() -> None:
    memory = RedisShortTermMemory(None)
    short_term_memory_service.initialize_service(memory)
    memory.add_message_at("session-2", "user", "original", "2026-08-09T00:00:00+00:00", turn_id="request-abcdefgh")

    with pytest.raises(RequestConflictError):
        chat_service._prepare_turn(
            ChatRequest(message="changed", session_id="session-2", request_id="request-abcdefgh")
        )


def test_reusing_request_id_in_another_session_is_rejected() -> None:
    memory = RedisShortTermMemory(None)
    short_term_memory_service.initialize_service(memory)
    memory.add_message_at(
        "original-session",
        "assistant",
        "completed",
        "2026-08-09T00:00:00+00:00",
        turn_id="request-crosssess",
    )

    with pytest.raises(RequestConflictError):
        chat_service._prepare_turn(
            ChatRequest(message="same", session_id="other-session", request_id="request-crosssess")
        )


def test_pii_is_accepted_as_customer_service_business_input() -> None:
    request = ChatRequest(
        message="张伟 18060815554 文艺路9号南京邮电大学仙林校区东门",
        request_id="request-sensitive1",
    )

    chat_service._validate_turn_request(request)


def test_chat_request_rejects_blank_and_oversized_messages() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 8001)


def test_malformed_compliance_output_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        compliance_checker_node,
        "invoke_llm",
        lambda *_args, **_kwargs: SimpleNamespace(content='{"passed": "yes", "risk_level": "low"}'),
    )

    result = compliance_checker_node.llm_check("ordinary customer service answer")

    assert result.passed is False
    assert result.risk_level == "high"


def test_partial_cache_cannot_shadow_complete_archive_history() -> None:
    archive = FakeArchive()
    for index in range(4):
        archive.add_message("session-3", "user", f"message-{index}", f"2026-08-09T00:00:0{index}+00:00")
    cache = RedisShortTermMemory(None, max_turns=10)
    cache.restore_history("session-3", archive.get_history("session-3", 2), total_count=2)
    memory = PersistentConversationMemory(cache, archive)

    history = memory.get_history("session-3", last_n=4)

    assert [item["content"] for item in history] == [
        "message-0",
        "message-1",
        "message-2",
        "message-3",
    ]


class RerankingEmbedding:
    def embed_query(self, _query: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "policy" in document else [0.0, 1.0] for document in documents]


class FailingEmbedding:
    def embed_query(self, _query: str) -> list[float]:
        raise TimeoutError("embedding unavailable")


def test_hybrid_rag_reranks_and_embedding_failure_degrades(tmp_path) -> None:
    reranked = LocalKnowledgeStore(tmp_path / "reranked", embedding_provider=RerankingEmbedding())
    reranked.add_document("refund refund process", "process.md")
    reranked.add_document("refund policy timing", "policy.md")

    result = reranked.search("refund", top_k=2)

    assert result[0]["source"] == "policy.md"
    assert reranked.health_status()["status"] == "ready"

    degraded = LocalKnowledgeStore(tmp_path / "degraded", embedding_provider=FailingEmbedding())
    degraded.add_document("refund refund process", "process.md")
    degraded.add_document("refund policy timing", "policy.md")
    fallback = degraded.search("refund", top_k=2)

    assert fallback[0]["source"] == "process.md"
    assert degraded.health_status()["status"] == "degraded"
    assert degraded.search("weather on mars", top_k=2) == []


def test_mcp_schema_blocks_invalid_arguments_before_handler() -> None:
    server = MCPToolServer()
    called = False

    @server.register(
        "bounded",
        "bounded integer",
        {"type": "object", "properties": {"top_k": {"type": "integer", "minimum": 1, "maximum": 5}}},
        effect="read",
        supports_idempotency=False,
    )
    async def bounded(top_k: int = 1, _trusted_context=None):
        nonlocal called
        called = True
        return top_k

    result = asyncio.run(server.call_tool("bounded", {"top_k": 99, "unexpected": True}))

    assert result.success is False
    assert result.error_code == "tool.validation"
    assert called is False


def test_mcp_call_log_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server_module,
        "get_settings",
        lambda: SimpleNamespace(tool_call_log_limit=10),
    )
    server = MCPToolServer()

    @server.register(
        "echo",
        "echo",
        {"type": "object", "properties": {}},
        effect="read",
        supports_idempotency=False,
    )
    async def echo(_trusted_context=None):
        return "ok"

    async def exercise() -> None:
        for _ in range(12):
            await server.call_tool("echo", {})

    asyncio.run(exercise())

    assert len(server.get_call_log(last_n=100)) == 10


def test_mcp_registration_rejects_handler_without_trusted_context() -> None:
    server = MCPToolServer()

    async def incompatible_handler(value: str):
        return value

    with pytest.raises(TypeError, match="_trusted_context"):
        server.register(
            "incompatible",
            "missing trusted context",
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
            effect="read",
            supports_idempotency=False,
        )(incompatible_handler)

    @server.register(
        "kwargs_handler",
        "trusted context through kwargs",
        {"type": "object", "properties": {}},
        effect="read",
        supports_idempotency=False,
    )
    async def kwargs_handler(**kwargs):
        return kwargs

    assert server.get_tool("kwargs_handler") is not None


def test_completed_agent_runs_expire_from_local_registry(monkeypatch) -> None:
    monkeypatch.setattr(conversation_archive_service, "instance", None)
    registry = AgentRunRegistry()
    registry._ttl_seconds = 1
    run = registry.begin("expired-session", "expired-turn")
    registry.finish(run, "completed")
    run.completed_at = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()

    assert registry.get("expired-session") is None


class QueryCountingClient:
    def __init__(self) -> None:
        self.calls = 0

    def execute_query(self, sql, args=None, fetch_one=False):
        self.calls += 1
        if "FROM cs_agent_runs" in sql:
            return True, [
                {
                    "turn_id": "turn-1",
                    "session_id": "session-1",
                    "status": "completed",
                    "stop_requested": 0,
                    "started_at": "2026-08-09T00:00:00+00:00",
                    "completed_at": "2026-08-09T00:00:01+00:00",
                },
                {
                    "turn_id": "turn-2",
                    "session_id": "session-1",
                    "status": "completed",
                    "stop_requested": 0,
                    "started_at": "2026-08-09T00:00:02+00:00",
                    "completed_at": "2026-08-09T00:00:03+00:00",
                },
            ]
        return True, [
            {
                "turn_id": "turn-1",
                "sequence_no": 0,
                "step_type": "thinking",
                "status": "success",
                "title": "step",
                "payload_json": "{}",
                "created_at": "2026-08-09T00:00:00+00:00",
            }
        ]


def test_agent_run_listing_batches_step_query() -> None:
    client = QueryCountingClient()
    archive = MySQLConversationArchive.__new__(MySQLConversationArchive)
    archive.mysql_client = client
    archive.user_id = "local-user"
    archive._ready = True

    runs = archive.list_runs("session-1", limit=100)

    assert len(runs) == 2
    assert client.calls == 2


def test_history_context_does_not_include_current_input() -> None:
    result = history_fusion_node({
        "raw_query": "current-input",
        "conversation_context": {
            "summary": "",
            "recent_messages": [{"role": "user", "content": "prior"}],
            "memories": [],
        },
    })

    messages = result["conversation_context"]["recent_messages"]
    assert all(message["content"] != "current-input" for message in messages)
    assert messages[0]["content"] == "prior"
