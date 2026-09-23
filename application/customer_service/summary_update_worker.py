from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from inspect import Parameter, signature

from domain.customer_service_agent.memory.models import MemoryOutboxEventType
from pkg.log.logger import get_logger
from pkg.telemetry import memory_metrics, normalize_error

logger = get_logger()


class SummaryUpdateWorker:
    def __init__(
        self,
        repository,
        summary_service,
        *,
        worker_id: str | None = None,
        lease_seconds: int = 60,
        max_attempts: int = 5,
        poll_seconds: float = 1.0,
        summary_cache=None,
        summary_eligible_turns: int = 20,
        summary_increment_turns: int = 8,
        summary_increment_chars: int = 0,
    ) -> None:
        self.repository = repository
        self.summary_service = summary_service
        self.worker_id = worker_id or f"summary-update-{uuid.uuid4().hex[:12]}"
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.poll_seconds = poll_seconds
        self.summary_cache = summary_cache
        self.summary_eligible_turns = summary_eligible_turns
        self.summary_increment_turns = summary_increment_turns
        self.summary_increment_chars = summary_increment_chars
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_success_at: datetime | None = None
        self._last_error_code: str | None = None

    def run_once(self, *, now: datetime | None = None) -> int:
        current = now or datetime.now(timezone.utc)
        events = self.repository.claim_outbox(
            [MemoryOutboxEventType.SUMMARY_UPDATE],
            worker_id=self.worker_id,
            limit=10,
            lease_seconds=self.lease_seconds,
            now=current,
        )
        memory_metrics.record_worker("summary", "claimed", len(events))
        completed = 0
        for event in events:
            try:
                self._process(event)
                self.repository.complete_outbox(str(event.event_id), current)
                completed += 1
                self._last_success_at = current
                self._last_error_code = None
                memory_metrics.record_worker("summary", "completed")
            except Exception as exc:
                error = normalize_error(exc)
                delay = min(2 ** max(event.attempts, 0), 300)
                self.repository.retry_outbox(
                    str(event.event_id),
                    error_code=str(error["error_code"]),
                    available_at=current + timedelta(seconds=delay),
                    max_attempts=self.max_attempts,
                )
                self._last_error_code = str(error["error_code"])
                outcome = "dead" if event.attempts + 1 >= self.max_attempts else "retry"
                memory_metrics.record_worker("summary", outcome)
                logger.warning(
                    "Memory summary update failed event_id={} error_type={} error_code={}",
                    event.event_id,
                    error["error_type"],
                    error["error_code"],
                )
        return completed

    def _process(self, event) -> None:
        user_id = str(event.payload["user_id"])
        session_id = str(event.payload["session_id"])
        previous = self.repository.get_latest_summary(user_id, session_id)
        after_message_id = previous.covers_until_message_id if previous else 0
        messages = self.repository.get_messages_after(
            user_id,
            session_id,
            after_message_id,
            limit=1000,
        )
        if not messages:
            return
        user_turns = sum(1 for item in messages if item.get("role") == "user")
        incremental_chars = sum(len(str(item.get("content", ""))) for item in messages)
        if previous is None and user_turns < self.summary_eligible_turns:
            return
        if previous is not None and user_turns < self.summary_increment_turns:
            if self.summary_increment_chars <= 0 or incremental_chars < self.summary_increment_chars:
                return
        summary = self.summary_service.build(
            user_id=user_id,
            session_id=session_id,
            previous=previous,
            new_messages=messages,
        )
        self.repository.save_summary(summary)
        cache_summary = getattr(self.summary_cache, "cache_session_summary", None)
        if callable(cache_summary):
            try:
                _call_user_scoped(
                    cache_summary,
                    session_id,
                    summary.model_dump(mode="json"),
                    user_id=user_id,
                )
            except Exception as exc:
                logger.warning(
                    "Memory summary cache update failed error_type={}",
                    type(exc).__name__,
                )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_forever,
            name="memory-summary-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def health_status(self) -> dict[str, object]:
        running = bool(self._thread is not None and self._thread.is_alive())
        return {
            "status": "ready" if running else "stopped",
            "running": running,
            "worker_id": self.worker_id,
            "last_success_at": self._last_success_at.isoformat() if self._last_success_at else None,
            "last_error_code": self._last_error_code,
        }

    def _run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as exc:
                error = normalize_error(exc)
                self._last_error_code = str(error["error_code"])
                memory_metrics.record_worker("summary", "loop_error")
                logger.warning(
                    "Memory summary worker loop failed error_type={} error_code={}",
                    error["error_type"],
                    error["error_code"],
                )
            self._stop.wait(self.poll_seconds)


def _call_user_scoped(method, *args, user_id: str):
    try:
        parameters = signature(method).parameters.values()
    except (TypeError, ValueError):
        return method(*args, user_id=user_id)
    supports_scope = any(
        parameter.name == "user_id" or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if supports_scope:
        return method(*args, user_id=user_id)
    return method(*args)
