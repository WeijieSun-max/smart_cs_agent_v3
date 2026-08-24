from __future__ import annotations

import asyncio
from collections import Counter, deque
from contextlib import asynccontextmanager

from pkg.config.settings import get_settings


class TurnAdmissionController:
    """Bounded round-robin queue with one active turn per user."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._queue: deque[tuple[str, object]] = deque()
        self._queued = Counter()
        self._active: set[str] = set()

    @asynccontextmanager
    async def slot(self, user_id: str):
        token = object()
        settings = get_settings()
        async with self._condition:
            if self._queued[user_id] >= settings.per_user_queue_limit:
                raise RuntimeError("user turn queue is full")
            if len(self._queue) >= settings.llm_queue_capacity:
                raise RuntimeError("agent admission queue is full")
            self._queue.append((user_id, token)); self._queued[user_id] += 1
            try:
                async with asyncio.timeout(settings.session_lock_timeout_seconds):
                    while not self._can_run(user_id, token):
                        await self._condition.wait()
            except BaseException:
                self._remove(user_id, token)
                self._condition.notify_all()
                raise
            self._remove(user_id, token); self._active.add(user_id)
        try:
            yield
        finally:
            async with self._condition:
                self._active.discard(user_id); self._condition.notify_all()

    def _can_run(self, user_id: str, token: object) -> bool:
        if user_id in self._active:
            return False
        # First queued request for each user participates in FIFO round robin.
        for queued_user, queued_token in self._queue:
            if queued_user == user_id:
                return queued_token is token
            if queued_user not in self._active:
                return False
        return False

    def _remove(self, user_id: str, token: object) -> None:
        try: self._queue.remove((user_id, token))
        except ValueError: return
        self._queued[user_id] -= 1
        if self._queued[user_id] <= 0: del self._queued[user_id]


turn_admission = TurnAdmissionController()
