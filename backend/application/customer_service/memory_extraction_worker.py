from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone

from domain.customer_service_agent.memory.models import (
    MemoryItem,
    MemoryOutboxEventType,
    MemorySource,
    MemoryStatus,
    MemoryType,
)
from pkg.log.logger import get_logger
from pkg.telemetry import memory_metrics, normalize_error

logger = get_logger()


class MemoryExtractionWorker:
    def __init__(
        self,
        repository,
        extraction_service,
        *,
        worker_id: str | None = None,
        lease_seconds: int = 60,
        max_attempts: int = 5,
        poll_seconds: float = 1.0,
        extraction_increment_turns: int = 1,
    ) -> None:
        self.repository = repository
        self.extraction_service = extraction_service
        self.worker_id = worker_id or f"memory-extract-{uuid.uuid4().hex[:12]}"
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.poll_seconds = poll_seconds
        self.extraction_increment_turns = extraction_increment_turns
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_success_at: datetime | None = None
        self._last_error_code: str | None = None

    def run_once(self, *, now: datetime | None = None) -> int:
        current = now or datetime.now(timezone.utc)
        events = self.repository.claim_outbox(
            [MemoryOutboxEventType.MEMORY_EXTRACT],
            worker_id=self.worker_id,
            limit=10,
            lease_seconds=self.lease_seconds,
            now=current,
        )
        memory_metrics.record_worker("extraction", "claimed", len(events))
        completed = 0
        for event in events:
            try:
                self._process(event, current)
                self.repository.complete_outbox(str(event.event_id), current)
                completed += 1
                self._last_success_at = current
                self._last_error_code = None
                memory_metrics.record_worker("extraction", "completed")
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
                memory_metrics.record_worker("extraction", outcome)
                logger.warning(
                    "Memory extraction failed event_id={} error_type={} error_code={}",
                    event.event_id,
                    error["error_type"],
                    error["error_code"],
                )
        return completed

    def _process(self, event, now: datetime) -> None:
        user_id = str(event.payload["user_id"])
        session_id = str(event.payload["session_id"])
        turn_id = str(event.payload["turn_id"])
        checkpoint = self.repository.get_extraction_checkpoint(user_id, session_id)
        messages = self.repository.get_messages_after(user_id, session_id, checkpoint, limit=1000)
        if not messages:
            return
        user_turns = sum(1 for item in messages if item.get("role") == "user")
        if user_turns < self.extraction_increment_turns:
            return
        previous = self.repository.get_latest_summary(user_id, session_id)
        candidates = self.extraction_service.extract(
            summary_text=previous.summary_text if previous else "",
            messages=messages,
            now=now,
        )
        items: list[MemoryItem] = []
        sources: list[MemorySource] = []
        superseded_ids: list[str] = []
        merged = 0
        for candidate in candidates:
            existing = self.repository.find_active_by_key(user_id, candidate.memory_type, candidate.memory_key)
            if existing is not None and existing.content == candidate.content:
                sources.append(self._source(existing.memory_id, session_id, turn_id, messages, now))
                merged += 1
                continue
            if existing is not None and candidate.confidence < existing.confidence:
                continue
            item = MemoryItem(
                user_id=user_id,
                memory_type=candidate.memory_type,
                memory_key=candidate.memory_key,
                content=candidate.content,
                structured_data=candidate.structured_data,
                confidence=candidate.confidence,
                status=MemoryStatus.ACTIVE,
                version=(existing.version + 1) if existing else 1,
                supersedes_id=existing.memory_id if existing else None,
                valid_from=now,
                expires_at=self._expires_at(candidate, now),
                created_at=now,
                updated_at=now,
            )
            items.append(item)
            sources.append(self._source(item.memory_id, session_id, turn_id, messages, now))
            if existing is not None:
                superseded_ids.append(str(existing.memory_id))
        last_message_id = max(int(item["message_id"]) for item in messages)
        applied = self.repository.apply_memory_extraction(
            user_id=user_id,
            session_id=session_id,
            expected_last_message_id=checkpoint,
            last_message_id=last_message_id,
            items=items,
            sources=sources,
            superseded_ids=superseded_ids,
            updated_at=now,
        )
        if not applied:
            raise RuntimeError("memory extraction checkpoint conflict")
        memory_metrics.record_extraction("created", len(items))
        memory_metrics.record_extraction("merged", merged)
        memory_metrics.record_extraction("superseded", len(superseded_ids))

    def _expires_at(self, candidate, now: datetime) -> datetime:
        expires_at = getattr(self.extraction_service, "expires_at", None)
        if callable(expires_at):
            return expires_at(candidate, now)
        days = {
            MemoryType.EPISODE: 180,
            MemoryType.PREFERENCE: 365,
            MemoryType.FACT: 180,
            MemoryType.TASK: 90,
        }[candidate.memory_type]
        return now + timedelta(days=days)

    @staticmethod
    def _source(memory_id, session_id: str, turn_id: str, messages: list[dict], now: datetime) -> MemorySource:
        return MemorySource(
            memory_id=memory_id,
            session_id=session_id,
            turn_id=turn_id,
            message_id=max(int(item["message_id"]) for item in messages),
            source_kind="conversation",
            created_at=now,
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, name="memory-extraction-worker", daemon=True)
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
                memory_metrics.record_worker("extraction", "loop_error")
                logger.warning(
                    "Memory extraction worker loop failed error_type={} error_code={}",
                    error["error_type"],
                    error["error_code"],
                )
            self._stop.wait(self.poll_seconds)
