from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone

from domain.customer_service_agent.memory.models import (
    MemoryOutboxEventType,
    VectorRecord,
)
from pkg.log.logger import get_logger
from pkg.telemetry import memory_metrics, normalize_error

logger = get_logger()


class MemoryIndexWorker:
    def __init__(
        self,
        repository,
        vector_index,
        embedder,
        *,
        worker_id: str | None = None,
        lease_seconds: int = 60,
        max_attempts: int = 5,
        poll_seconds: float = 1.0,
    ) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.embedder = embedder
        self.worker_id = worker_id or f"memory-index-{uuid.uuid4().hex[:12]}"
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_success_at: datetime | None = None
        self._last_error_code: str | None = None

    def run_once(self, *, now: datetime | None = None) -> int:
        current = now or datetime.now(timezone.utc)
        events = self.repository.claim_outbox(
            [MemoryOutboxEventType.INDEX_UPSERT, MemoryOutboxEventType.INDEX_DELETE],
            worker_id=self.worker_id,
            limit=20,
            lease_seconds=self.lease_seconds,
            now=current,
        )
        memory_metrics.record_worker("index", "claimed", len(events))
        completed = 0
        for event in events:
            try:
                self._process(event, current)
                self.repository.complete_outbox(str(event.event_id), current)
                completed += 1
                self._last_success_at = current
                self._last_error_code = None
                memory_metrics.record_worker("index", "completed")
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
                memory_metrics.record_worker("index", outcome)
                logger.warning(
                    "Memory index update failed event_id={} error_type={} error_code={}",
                    event.event_id,
                    error["error_type"],
                    error["error_code"],
                )
        return completed

    def _process(self, event, now: datetime) -> None:
        memory_id = str(event.payload.get("memory_id") or event.aggregate_id)
        if event.event_type == MemoryOutboxEventType.INDEX_DELETE:
            self.vector_index.delete([memory_id])
            return
        user_id = str(event.payload["user_id"])
        items = self.repository.get_items_by_ids(user_id, [memory_id], now)
        if not items:
            self.vector_index.delete([memory_id])
            return
        vectors = self.embedder.embed_documents([item.content for item in items])
        self.vector_index.upsert([
            VectorRecord(
                memory_id=item.memory_id,
                vector=vector,
                user_id=item.user_id,
                memory_type=item.memory_type,
                status=item.status,
                confidence=item.confidence,
                updated_at=item.updated_at,
                expires_at=item.expires_at,
            )
            for item, vector in zip(items, vectors)
        ])

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, name="memory-index-worker", daemon=True)
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
                memory_metrics.record_worker("index", "loop_error")
                logger.warning(
                    "Memory index worker loop failed error_type={} error_code={}",
                    error["error_type"],
                    error["error_code"],
                )
            self._stop.wait(self.poll_seconds)
