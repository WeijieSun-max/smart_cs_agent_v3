from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Optional

from redis import Redis

from domain.customer_service_agent.interfaces.i_short_term_memory import IShortTermMemory
from pkg.telemetry import record_fallback
from pkg.security import get_local_user_id


class RedisShortTermMemory(IShortTermMemory):
    def __init__(
        self,
        redis_client: Optional[Redis],
        ttl_seconds: int = 1800,
        max_turns: int = 20,
        user_id: str | None = None,
    ):
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self.user_id = user_id or get_local_user_id()
        self._fallback_store: dict[str, list[dict[str, str]]] = {}
        self._fallback_sessions: dict[str, dict[str, object]] = {}
        self._fallback_summaries: dict[str, dict[str, object]] = {}
        self._fallback_lock = threading.RLock()

    def switch_user(self, user_id: str) -> None:
        if user_id == self.user_id:
            return
        with self._fallback_lock:
            self.user_id = user_id
            self._fallback_store.clear()
            self._fallback_sessions.clear()
            self._fallback_summaries.clear()

    def _key(self, session_id: str) -> str:
        return f"smartcs:{self.user_id}:short_term:{session_id}"

    def _session_key(self, session_id: str) -> str:
        return f"smartcs:{self.user_id}:session:{session_id}"

    def _turn_key(self, turn_id: str) -> str:
        return f"smartcs:{self.user_id}:turn:{turn_id}:messages"

    def _sessions_key(self) -> str:
        return f"smartcs:{self.user_id}:sessions"

    def _summary_key(self, session_id: str) -> str:
        return f"smartcs:{self.user_id}:memory_summary:{session_id}"

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _title_from_message(content: str) -> str:
        title = " ".join(content.strip().split())
        return title[:40] + ("…" if len(title) > 40 else "") or "新会话"

    def add_message(self, session_id: str, role: str, content: str, turn_id: str | None = None) -> None:
        self.add_message_at(session_id, role, content, self._now().isoformat(), turn_id=turn_id)

    def add_message_at(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: str,
        *,
        turn_id: str | None = None,
    ) -> None:
        try:
            now = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            now = self._now()
            timestamp = now.isoformat()
        message = {"session_id": session_id, "role": role, "content": content, "timestamp": timestamp}
        if turn_id:
            message["turn_id"] = turn_id
        if self.redis_client is not None:
            key = self._key(session_id)
            session_key = self._session_key(session_id)
            if not self.redis_client.exists(session_key):
                self.create_session(session_id)
            current = self.redis_client.hgetall(session_key)
            updates: dict[str, object] = {
                "updated_at": now.isoformat(),
                "message_count": int(current.get("message_count", 0)) + 1,
            }
            if role == "user" and int(current.get("message_count", 0)) == 0:
                updates["title"] = self._title_from_message(content)
            with self.redis_client.pipeline() as pipeline:
                pipeline.rpush(key, self._serialize_message(message))
                pipeline.ltrim(key, -self.max_turns, -1)
                pipeline.expire(key, self.ttl_seconds)
                pipeline.hset(session_key, mapping=updates)
                pipeline.expire(session_key, self.ttl_seconds)
                pipeline.zadd(self._sessions_key(), {session_id: now.timestamp()})
                if turn_id:
                    pipeline.hset(self._turn_key(turn_id), role, self._serialize_message(message))
                    pipeline.expire(self._turn_key(turn_id), self.ttl_seconds)
                pipeline.execute()
            return
        record_fallback("redis")
        with self._fallback_lock:
            session = self._fallback_sessions.get(session_id)
            if session is None:
                session = self.create_session(session_id)
            if role == "user" and int(session["message_count"]) == 0:
                session["title"] = self._title_from_message(content)
            session["updated_at"] = now.isoformat()
            session["message_count"] = int(session["message_count"]) + 1
            self._fallback_store.setdefault(session_id, []).append(message)
            self._fallback_store[session_id] = self._fallback_store[session_id][-self.max_turns:]

    def restore_history(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        *,
        total_count: int | None = None,
        session_metadata: dict[str, object] | None = None,
    ) -> None:
        if not messages:
            return
        cached_messages = messages[-self.max_turns:]
        first_user = next((message for message in messages if message.get("role") == "user"), messages[0])
        created_at = messages[0].get("timestamp") or self._now().isoformat()
        updated_at = messages[-1].get("timestamp") or created_at
        session: dict[str, object] = {
            "id": session_id,
            "title": self._title_from_message(first_user.get("content", "")),
            "agent_id": "general",
            "created_at": created_at,
            "updated_at": updated_at,
            "favorite": False,
            "message_count": total_count if total_count is not None else len(messages),
        }
        if session_metadata:
            session.update({
                "title": session_metadata.get("title", session["title"]),
                "agent_id": session_metadata.get("agent_id", session["agent_id"]),
                "created_at": session_metadata.get("created_at", session["created_at"]),
                "updated_at": session_metadata.get("updated_at", session["updated_at"]),
                "favorite": bool(session_metadata.get("favorite", session["favorite"])),
            })
        if self.redis_client is not None:
            history_key = self._key(session_id)
            session_key = self._session_key(session_id)
            try:
                score = datetime.fromisoformat(updated_at.replace("Z", "+00:00")).timestamp()
            except ValueError:
                score = self._now().timestamp()
            with self.redis_client.pipeline() as pipeline:
                pipeline.delete(history_key, session_key)
                pipeline.rpush(history_key, *(self._serialize_message(message) for message in cached_messages))
                pipeline.expire(history_key, self.ttl_seconds)
                pipeline.hset(session_key, mapping=self._serialize_session(session))
                pipeline.expire(session_key, self.ttl_seconds)
                pipeline.zadd(self._sessions_key(), {session_id: score})
                for message in cached_messages:
                    cached_turn_id = message.get("turn_id")
                    cached_role = message.get("role")
                    if cached_turn_id and cached_role:
                        pipeline.hset(
                            self._turn_key(cached_turn_id),
                            cached_role,
                            self._serialize_message(message),
                        )
                        pipeline.expire(self._turn_key(cached_turn_id), self.ttl_seconds)
                pipeline.execute()
            return
        with self._fallback_lock:
            self._fallback_store[session_id] = [dict(message) for message in cached_messages]
            self._fallback_sessions[session_id] = session

    def get_history(self, session_id: str, last_n: int | None = None) -> list[dict[str, str]]:
        n = last_n or self.max_turns
        if self.redis_client is not None:
            raw = self.redis_client.lrange(self._key(session_id), -n, -1)
            return [self._deserialize_message(item) for item in raw]
        record_fallback("redis")
        with self._fallback_lock:
            return list(self._fallback_store.get(session_id, [])[-n:])

    def get_message_by_turn(self, turn_id: str, role: str) -> dict[str, str] | None:
        if not turn_id:
            return None
        if self.redis_client is not None:
            raw = self.redis_client.hget(self._turn_key(turn_id), role)
            return self._deserialize_message(raw) if raw else None
        with self._fallback_lock:
            for messages in self._fallback_store.values():
                for message in reversed(messages):
                    if message.get("turn_id") == turn_id and message.get("role") == role:
                        return dict(message)
        return None

    def get_session_summary(self, session_id: str) -> dict[str, object] | None:
        if self.redis_client is not None:
            raw = self.redis_client.get(self._summary_key(session_id))
            if not raw:
                return None
            text = raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else raw
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        with self._fallback_lock:
            summary = self._fallback_summaries.get(session_id)
            return dict(summary) if summary else None

    def cache_session_summary(self, session_id: str, summary: dict[str, object]) -> None:
        if self.redis_client is not None:
            self.redis_client.set(
                self._summary_key(session_id),
                json.dumps(summary, ensure_ascii=False, separators=(",", ":"), default=str),
                ex=self.ttl_seconds,
            )
            return
        with self._fallback_lock:
            self._fallback_summaries[session_id] = dict(summary)

    def delete_session_summary(self, session_id: str) -> None:
        if self.redis_client is not None:
            self.redis_client.delete(self._summary_key(session_id))
            return
        with self._fallback_lock:
            self._fallback_summaries.pop(session_id, None)

    def get_session(self, session_id: str) -> dict[str, object] | None:
        if self.redis_client is not None:
            raw = self.redis_client.hgetall(self._session_key(session_id))
            return self._deserialize_session(raw) if raw else None
        with self._fallback_lock:
            session = self._fallback_sessions.get(session_id)
            return dict(session) if session else None

    def get_context_window(self, session_id: str, max_chars: int = 4000) -> str:
        history = self.get_history(session_id)
        parts = []
        length = 0
        for msg in reversed(history):
            text = f"{msg['role']}: {msg['content']}"
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
        now = self._now()
        session: dict[str, object] = {
            "id": session_id,
            "title": title.strip() or "新会话",
            "agent_id": agent_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "favorite": False,
            "message_count": 0,
        }
        if self.redis_client is not None:
            key = self._session_key(session_id)
            existing = self.redis_client.hgetall(key)
            if existing:
                return self._deserialize_session(existing)
            with self.redis_client.pipeline() as pipeline:
                pipeline.hset(key, mapping=self._serialize_session(session))
                pipeline.expire(key, self.ttl_seconds)
                pipeline.zadd(self._sessions_key(), {session_id: now.timestamp()})
                pipeline.execute()
            return session
        record_fallback("redis")
        with self._fallback_lock:
            return self._fallback_sessions.setdefault(session_id, session)

    def list_sessions(self) -> list[dict[str, object]]:
        if self.redis_client is not None:
            self._backfill_session_index()
            session_ids = self.redis_client.zrevrange(self._sessions_key(), 0, -1)
            sessions: list[dict[str, object]] = []
            stale_ids: list[str] = []
            for session_id in session_ids:
                raw = self.redis_client.hgetall(self._session_key(session_id))
                if raw:
                    sessions.append(self._deserialize_session(raw))
                else:
                    stale_ids.append(session_id)
            if stale_ids:
                self.redis_client.zrem(self._sessions_key(), *stale_ids)
            return sessions
        record_fallback("redis")
        with self._fallback_lock:
            return sorted(
                (dict(session) for session in self._fallback_sessions.values()),
                key=lambda session: str(session["updated_at"]),
                reverse=True,
            )

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        favorite: bool | None = None,
    ) -> dict[str, object] | None:
        now = self._now()
        if self.redis_client is not None:
            key = self._session_key(session_id)
            raw = self.redis_client.hgetall(key)
            if not raw:
                return None
            updates: dict[str, object] = {}
            if title is not None:
                updates["title"] = title.strip() or "新会话"
            if favorite is not None:
                updates["favorite"] = "1" if favorite else "0"
            if updates:
                updates["updated_at"] = now.isoformat()
                with self.redis_client.pipeline() as pipeline:
                    pipeline.hset(key, mapping=updates)
                    pipeline.expire(key, self.ttl_seconds)
                    pipeline.zadd(self._sessions_key(), {session_id: now.timestamp()})
                    pipeline.execute()
                raw.update({name: str(value) for name, value in updates.items()})
            return self._deserialize_session(raw)
        record_fallback("redis")
        with self._fallback_lock:
            session = self._fallback_sessions.get(session_id)
            if session is None:
                return None
            if title is not None:
                session["title"] = title.strip() or "新会话"
            if favorite is not None:
                session["favorite"] = favorite
            if title is not None or favorite is not None:
                session["updated_at"] = now.isoformat()
            return dict(session)

    def delete_session(self, session_id: str) -> bool:
        if self.redis_client is not None:
            removed = self.redis_client.delete(
                self._key(session_id),
                self._session_key(session_id),
                self._summary_key(session_id),
            )
            self.redis_client.zrem(self._sessions_key(), session_id)
            return bool(removed)
        record_fallback("redis")
        with self._fallback_lock:
            existed = session_id in self._fallback_sessions or session_id in self._fallback_store
            self._fallback_sessions.pop(session_id, None)
            self._fallback_store.pop(session_id, None)
            self._fallback_summaries.pop(session_id, None)
            return existed

    def _backfill_session_index(self) -> None:
        """Expose histories created before session metadata was introduced."""
        assert self.redis_client is not None
        prefix = f"smartcs:{self.user_id}:short_term:"
        for history_key in self.redis_client.scan_iter(match=f"{prefix}*", count=100):
            session_id = history_key[len(prefix):]
            session_key = self._session_key(session_id)
            if self.redis_client.exists(session_key):
                continue
            raw_messages = self.redis_client.lrange(history_key, 0, -1)
            if not raw_messages:
                continue
            messages = []
            for item in raw_messages:
                try:
                    messages.append(self._deserialize_message(item))
                except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                    continue
            if not messages:
                continue
            first_user = next((item for item in messages if item.get("role") == "user"), messages[0])
            created_at = messages[0].get("timestamp") or self._now().isoformat()
            updated_at = messages[-1].get("timestamp") or created_at
            session: dict[str, object] = {
                "id": session_id,
                "title": self._title_from_message(str(first_user.get("content", ""))),
                "agent_id": "general",
                "created_at": created_at,
                "updated_at": updated_at,
                "favorite": False,
                "message_count": len(messages),
            }
            try:
                score = datetime.fromisoformat(str(updated_at)).timestamp()
            except ValueError:
                score = self._now().timestamp()
            ttl = self.redis_client.ttl(history_key)
            with self.redis_client.pipeline() as pipeline:
                pipeline.hset(session_key, mapping=self._serialize_session(session))
                if ttl > 0:
                    pipeline.expire(session_key, ttl)
                pipeline.zadd(self._sessions_key(), {session_id: score})
                pipeline.execute()

    @staticmethod
    def _serialize_session(session: dict[str, object]) -> dict[str, object]:
        return {
            **session,
            "favorite": "1" if session["favorite"] else "0",
        }

    @staticmethod
    def _serialize_message(message: dict[str, str]) -> str:
        """Keep CJK content as literal Unicode before redis-py encodes it as UTF-8."""
        return json.dumps(message, ensure_ascii=False)

    @staticmethod
    def _deserialize_message(raw: str | bytes) -> dict[str, str]:
        """Return text values for both current strings and legacy byte responses."""
        text = raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else raw
        return json.loads(text)

    @staticmethod
    def _deserialize_session(raw: dict[str, str]) -> dict[str, object]:
        return {
            "id": raw["id"],
            "title": raw.get("title", "新会话"),
            "agent_id": raw.get("agent_id", "general"),
            "created_at": raw.get("created_at", ""),
            "updated_at": raw.get("updated_at", ""),
            "favorite": raw.get("favorite", "0") in {"1", "true", "True"},
            "message_count": int(raw.get("message_count", 0)),
        }
