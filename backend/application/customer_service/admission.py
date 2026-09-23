from __future__ import annotations

import asyncio
from collections import Counter, deque
from contextlib import asynccontextmanager
from typing import Hashable

from pkg.config.settings import get_settings


SessionKey = tuple[str, Hashable]


class TurnAdmissionController:
    """Bound all active turns and serialize writes to the same user session."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._queue: deque[tuple[SessionKey, object]] = deque()
        self._queued_by_session: Counter[SessionKey] = Counter()
        self._active_sessions: set[SessionKey] = set()
        self._active_count = 0

    @asynccontextmanager
    async def slot(self, user_id: str, session_id: str | None = None):
        token = object()
        # New-session requests need distinct keys until their generated IDs exist.
        session_key: SessionKey = (user_id, session_id if session_id is not None else token)
        settings = get_settings()
        acquired = False

        async with self._condition:
            if self._can_start_immediately(session_key, settings.turn_max_concurrency):
                self._activate(session_key)
                acquired = True
            else:
                if self._queued_by_session[session_key] >= settings.per_session_queue_limit:
                    raise RuntimeError("session turn queue is full")
                if len(self._queue) >= settings.turn_queue_capacity:
                    raise RuntimeError("agent admission queue is full")
                self._queue.append((session_key, token))
                self._queued_by_session[session_key] += 1
                try:
                    async with asyncio.timeout(settings.turn_queue_timeout_seconds):
                        while not self._can_run(session_key, token, settings.turn_max_concurrency):
                            await self._condition.wait()
                except TimeoutError as exc:
                    self._remove(session_key, token)
                    self._condition.notify_all()
                    raise RuntimeError("agent admission queue timed out") from exc
                except BaseException:
                    self._remove(session_key, token)
                    self._condition.notify_all()
                    raise
                self._remove(session_key, token)
                self._activate(session_key)
                acquired = True

        try:
            yield
        finally:
            if acquired:
                async with self._condition:
                    self._active_sessions.discard(session_key)
                    self._active_count -= 1
                    self._condition.notify_all()

    def snapshot(self) -> dict[str, int]:
        """Return process-local pressure for health checks and tests."""

        return {
            "active": self._active_count,
            "queued": len(self._queue),
            "active_sessions": len(self._active_sessions),
        }

    def _can_start_immediately(self, session_key: SessionKey, max_concurrency: int) -> bool:
        if self._active_count >= max_concurrency or session_key in self._active_sessions:
            return False
        return not any(key not in self._active_sessions for key, _ in self._queue)

    def _can_run(self, session_key: SessionKey, token: object, max_concurrency: int) -> bool:
        if self._active_count >= max_concurrency or session_key in self._active_sessions:
            return False
        # A waiter behind an active session must not block unrelated sessions.
        for queued_key, queued_token in self._queue:
            if queued_key not in self._active_sessions:
                return queued_key == session_key and queued_token is token
        return False

    def _activate(self, session_key: SessionKey) -> None:
        self._active_sessions.add(session_key)
        self._active_count += 1

    def _remove(self, session_key: SessionKey, token: object) -> None:
        try:
            self._queue.remove((session_key, token))
        except ValueError:
            return
        self._queued_by_session[session_key] -= 1
        if self._queued_by_session[session_key] <= 0:
            del self._queued_by_session[session_key]


turn_admission = TurnAdmissionController()
