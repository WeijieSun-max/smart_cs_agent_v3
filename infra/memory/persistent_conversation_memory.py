from __future__ import annotations

from datetime import datetime, timezone

from domain.customer_service_agent.interfaces.i_conversation_archive import IConversationArchive
from domain.customer_service_agent.interfaces.i_short_term_memory import IShortTermMemory
from infra.memory.short_term_memory import RedisShortTermMemory
from pkg.exceptions.exception import StorageOperationError, StorageUnavailableError
from pkg.telemetry import record_fallback


class PersistentConversationMemory(IShortTermMemory):
    """Redis-backed hot memory with MySQL as the durable source of truth."""

    def __init__(self, cache: RedisShortTermMemory, archive: IConversationArchive) -> None:
        self.cache = cache
        self.archive = archive

    def _require_archive(self) -> None:
        if not self.archive.available:
            raise StorageUnavailableError()

    def add_message(self, session_id: str, role: str, content: str, turn_id: str | None = None) -> None:
        self._require_archive()
        timestamp = datetime.now(timezone.utc).isoformat()
        archived = (
            self.archive.add_message(session_id, role, content, timestamp, turn_id=turn_id)
            if turn_id
            else self.archive.add_message(session_id, role, content, timestamp)
        )
        if not archived:
            raise StorageOperationError()
        try:
            self.cache.add_message_at(session_id, role, content, timestamp, turn_id=turn_id)
        except Exception:
            record_fallback("redis")

    def complete_turn(
        self,
        session_id: str,
        content: str,
        turn_id: str,
        *,
        enqueue_memory: bool = False,
    ) -> None:
        self._require_archive()
        timestamp = datetime.now(timezone.utc).isoformat()
        archived = self.archive.complete_turn(
            session_id,
            content,
            timestamp,
            turn_id,
            enqueue_memory=enqueue_memory,
        )
        if not archived:
            raise StorageOperationError()
        try:
            self.cache.add_message_at(session_id, "assistant", content, timestamp, turn_id=turn_id)
        except Exception:
            record_fallback("redis")

    def get_message_by_turn(self, turn_id: str, role: str) -> dict[str, str] | None:
        self._require_archive()
        return self.archive.get_message_by_turn(turn_id, role)

    def get_session_summary(self, session_id: str) -> dict[str, object] | None:
        try:
            return self.cache.get_session_summary(session_id)
        except Exception:
            record_fallback("redis")
            return None

    def cache_session_summary(self, session_id: str, summary: dict[str, object]) -> None:
        try:
            self.cache.cache_session_summary(session_id, summary)
        except Exception:
            record_fallback("redis")

    def delete_session_summary(self, session_id: str) -> None:
        try:
            self.cache.delete_session_summary(session_id)
        except Exception:
            record_fallback("redis")

    def get_history(self, session_id: str, last_n: int | None = None) -> list[dict[str, str]]:
        self._require_archive()
        session = self.archive.get_session(session_id)
        if session is None:
            return []
        requested = last_n or self.cache.max_turns
        total_count = int(session.get("message_count", 0))
        history: list[dict[str, str]] = []
        cached_session: dict[str, object] | None = None
        try:
            history = self.cache.get_history(session_id, last_n=last_n)
            cached_session = self.cache.get_session(session_id)
        except Exception:
            record_fallback("redis")
        cache_limit_ok = requested <= self.cache.max_turns
        expected_cached = min(total_count, requested)
        cache_version_ok = (
            cached_session is not None
            and int(cached_session.get("message_count", -1)) == total_count
            and len(history) >= expected_cached
        )
        if cache_limit_ok and cache_version_ok:
            return history[-requested:]
        history = self.archive.get_history(session_id, requested)
        if history:
            try:
                self.cache.restore_history(
                    session_id,
                    history,
                    total_count=total_count,
                    session_metadata=session,
                )
            except Exception:
                record_fallback("redis")
        return history

    def get_context_window(self, session_id: str, max_chars: int = 4000) -> str:
        history = self.get_history(session_id)
        parts: list[str] = []
        length = 0
        for message in reversed(history):
            text = f"{message['role']}: {message['content']}"
            if length + len(text) > max_chars:
                break
            parts.insert(0, text)
            length += len(text)
        return "\n".join(parts)

    def create_session(
        self,
        session_id: str,
        title: str = "新会话",
        agent_id: str = "general",
    ) -> dict[str, object]:
        self._require_archive()
        archived = self.archive.create_session(session_id, title, agent_id)
        if archived is None:
            raise StorageOperationError()
        try:
            self.cache.create_session(session_id, title, agent_id)
        except Exception:
            record_fallback("redis")
        return archived

    def list_sessions(self) -> list[dict[str, object]]:
        self._require_archive()
        return self.archive.list_sessions()

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        favorite: bool | None = None,
    ) -> dict[str, object] | None:
        self._require_archive()
        archived = self.archive.update_session(session_id, title=title, favorite=favorite)
        if archived is None:
            return None
        try:
            cached = self.cache.update_session(session_id, title=title, favorite=favorite)
        except Exception:
            record_fallback("redis")
            cached = None
        if archived is not None and cached is None:
            try:
                self.cache.create_session(
                    session_id,
                    title=str(archived["title"]),
                    agent_id=str(archived["agent_id"]),
                )
                self.cache.update_session(session_id, title=title, favorite=favorite)
            except Exception:
                record_fallback("redis")
        return archived

    def delete_session(self, session_id: str) -> bool:
        self._require_archive()
        archived = self.archive.delete_session(session_id)
        if not archived:
            return False
        try:
            self.cache.delete_session(session_id)
        except Exception:
            record_fallback("redis")
        return True

    def migrate_cache_to_archive(self) -> int:
        # Cache data is not authoritative in persistence-first mode. Importing
        # it automatically could resurrect previously deleted or partial data.
        return 0
