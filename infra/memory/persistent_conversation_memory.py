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

    @staticmethod
    def _call_scoped(method, *args, user_id: str | None = None, **kwargs):
        # Older test/adapter implementations remain usable when no explicit
        # request scope is needed. Production request paths always pass user_id.
        if user_id is None:
            return method(*args, **kwargs)
        return method(*args, user_id=user_id, **kwargs)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        turn_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> None:
        self._require_archive()
        timestamp = datetime.now(timezone.utc).isoformat()
        archived = (
            self._call_scoped(
                self.archive.add_message,
                session_id,
                role,
                content,
                timestamp,
                turn_id=turn_id,
                user_id=user_id,
            )
            if turn_id
            else self._call_scoped(
                self.archive.add_message,
                session_id,
                role,
                content,
                timestamp,
                user_id=user_id,
            )
        )
        if not archived:
            raise StorageOperationError()
        try:
            self.cache.add_message_at(
                session_id,
                role,
                content,
                timestamp,
                turn_id=turn_id,
                user_id=user_id,
            )
        except Exception:
            record_fallback("redis")

    def complete_turn(
        self,
        session_id: str,
        content: str,
        turn_id: str,
        *,
        enqueue_memory: bool = False,
        user_id: str | None = None,
        fencing_token: int | None = None,
        lease_owner_id: str | None = None,
    ) -> None:
        self._require_archive()
        timestamp = datetime.now(timezone.utc).isoformat()
        lease_kwargs = {}
        if fencing_token is not None:
            lease_kwargs = {
                "fencing_token": fencing_token,
                "lease_owner_id": lease_owner_id,
            }
        archived = self._call_scoped(
            self.archive.complete_turn,
            session_id,
            content,
            timestamp,
            turn_id,
            enqueue_memory=enqueue_memory,
            user_id=user_id,
            **lease_kwargs,
        )
        if not archived:
            raise StorageOperationError()
        try:
            self.cache.add_message_at(
                session_id,
                "assistant",
                content,
                timestamp,
                turn_id=turn_id,
                user_id=user_id,
            )
        except Exception:
            record_fallback("redis")

    def get_message_by_turn(
        self,
        turn_id: str,
        role: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, str] | None:
        self._require_archive()
        return self._call_scoped(self.archive.get_message_by_turn, turn_id, role, user_id=user_id)

    def get_session_summary(self, session_id: str, *, user_id: str | None = None) -> dict[str, object] | None:
        try:
            return self.cache.get_session_summary(session_id, user_id=user_id)
        except Exception:
            record_fallback("redis")
            return None

    def cache_session_summary(
        self,
        session_id: str,
        summary: dict[str, object],
        *,
        user_id: str | None = None,
    ) -> None:
        try:
            self.cache.cache_session_summary(session_id, summary, user_id=user_id)
        except Exception:
            record_fallback("redis")

    def delete_session_summary(self, session_id: str, *, user_id: str | None = None) -> None:
        try:
            self.cache.delete_session_summary(session_id, user_id=user_id)
        except Exception:
            record_fallback("redis")

    def get_history(
        self,
        session_id: str,
        last_n: int | None = None,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, str]]:
        self._require_archive()
        session = self._call_scoped(self.archive.get_session, session_id, user_id=user_id)
        if session is None:
            return []
        requested = last_n or self.cache.max_turns
        total_count = int(session.get("message_count", 0))
        history: list[dict[str, str]] = []
        cached_session: dict[str, object] | None = None
        try:
            history = self.cache.get_history(session_id, last_n=last_n, user_id=user_id)
            cached_session = self.cache.get_session(session_id, user_id=user_id)
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
        history = self._call_scoped(self.archive.get_history, session_id, requested, user_id=user_id)
        if history:
            try:
                self.cache.restore_history(
                    session_id,
                    history,
                    total_count=total_count,
                    session_metadata=session,
                    user_id=user_id,
                )
            except Exception:
                record_fallback("redis")
        return history

    def get_context_window(
        self,
        session_id: str,
        max_chars: int = 4000,
        *,
        user_id: str | None = None,
    ) -> str:
        history = self.get_history(session_id, user_id=user_id)
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
        *,
        user_id: str | None = None,
    ) -> dict[str, object]:
        self._require_archive()
        archived = self._call_scoped(
            self.archive.create_session,
            session_id,
            title,
            agent_id,
            user_id=user_id,
        )
        if archived is None:
            raise StorageOperationError()
        try:
            self.cache.create_session(session_id, title, agent_id, user_id=user_id)
        except Exception:
            record_fallback("redis")
        return archived

    def list_sessions(self, *, user_id: str | None = None) -> list[dict[str, object]]:
        self._require_archive()
        return self._call_scoped(self.archive.list_sessions, user_id=user_id)

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        favorite: bool | None = None,
        user_id: str | None = None,
    ) -> dict[str, object] | None:
        self._require_archive()
        archived = self._call_scoped(
            self.archive.update_session,
            session_id,
            title=title,
            favorite=favorite,
            user_id=user_id,
        )
        if archived is None:
            return None
        try:
            cached = self.cache.update_session(
                session_id,
                title=title,
                favorite=favorite,
                user_id=user_id,
            )
        except Exception:
            record_fallback("redis")
            cached = None
        if archived is not None and cached is None:
            try:
                self.cache.create_session(
                    session_id,
                    title=str(archived["title"]),
                    agent_id=str(archived["agent_id"]),
                    user_id=user_id,
                )
                self.cache.update_session(
                    session_id,
                    title=title,
                    favorite=favorite,
                    user_id=user_id,
                )
            except Exception:
                record_fallback("redis")
        return archived

    def delete_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        self._require_archive()
        archived = self._call_scoped(self.archive.delete_session, session_id, user_id=user_id)
        if not archived:
            return False
        try:
            self.cache.delete_session(session_id, user_id=user_id)
        except Exception:
            record_fallback("redis")
        return True

    def migrate_cache_to_archive(self) -> int:
        # Cache data is not authoritative in persistence-first mode. Importing
        # it automatically could resurrect previously deleted or partial data.
        return 0
