from datetime import datetime, timezone
from types import SimpleNamespace

from application.customer_service.memory_extraction_worker import MemoryExtractionWorker
from domain.customer_service_agent.memory.models import (
    MemoryCandidate,
    MemoryOutboxEvent,
    MemoryOutboxEventType,
    MemoryType,
    SessionSummary,
)


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


class FakeRepository:
    def __init__(self) -> None:
        self.events = [MemoryOutboxEvent(
            event_type=MemoryOutboxEventType.TURN_COMPLETED,
            aggregate_id="turn-1",
            payload={"user_id": "user-1", "session_id": "session-1", "turn_id": "turn-1"},
            available_at=NOW,
            created_at=NOW,
        )]
        self.completed: list[str] = []
        self.retried: list[tuple[str, dict]] = []
        self.applied = []

    def claim_outbox(self, *args, **kwargs):
        events, self.events = self.events, []
        return events

    def get_latest_summary(self, user_id, session_id):
        return None

    def get_messages_after(self, user_id, session_id, after_message_id, limit=1000):
        return [
            {"message_id": 1, "role": "user", "content": "我偏好稳健产品", "turn_id": "turn-1"},
            {"message_id": 2, "role": "assistant", "content": "已了解", "turn_id": "turn-1"},
        ]

    def find_active_by_key(self, user_id, memory_type, memory_key):
        return None

    def apply_extraction(self, summary, items, sources, superseded_ids):
        self.applied.append((summary, items, sources, superseded_ids))

    def complete_outbox(self, event_id, processed_at):
        self.completed.append(str(event_id))

    def retry_outbox(self, event_id, **kwargs):
        self.retried.append((str(event_id), kwargs))


def test_worker_applies_summary_and_memory_once() -> None:
    repository = FakeRepository()
    summary = SessionSummary(
        user_id="user-1",
        session_id="session-1",
        version=1,
        summary_text="用户偏好稳健产品",
        structured_data={},
        covers_until_message_id=2,
        created_at=NOW,
        updated_at=NOW,
    )
    summary_service = SimpleNamespace(build=lambda **kwargs: summary)
    extraction_service = SimpleNamespace(extract=lambda **kwargs: [MemoryCandidate(
        memory_type=MemoryType.PREFERENCE,
        memory_key="preference.risk",
        content="用户偏好稳健产品",
        confidence=0.9,
    )])
    cached = []
    summary_cache = SimpleNamespace(cache_session_summary=lambda session_id, value: cached.append((session_id, value)))
    worker = MemoryExtractionWorker(
        repository,
        summary_service,
        extraction_service,
        worker_id="worker-1",
        summary_cache=summary_cache,
    )

    assert worker.run_once(now=NOW) == 1
    assert worker.run_once(now=NOW) == 0
    assert len(repository.applied) == 1
    assert len(repository.applied[0][1]) == 1
    assert len(repository.completed) == 1
    assert cached[0][0] == "session-1"
    assert cached[0][1]["version"] == 1


def test_worker_completes_event_without_summary_before_threshold() -> None:
    repository = FakeRepository()
    summary_service = SimpleNamespace(
        build=lambda **kwargs: (_ for _ in ()).throw(AssertionError("summary must not run"))
    )
    worker = MemoryExtractionWorker(
        repository,
        summary_service,
        SimpleNamespace(extract=lambda **kwargs: []),
        summary_eligible_turns=2,
    )

    assert worker.run_once(now=NOW) == 1
    assert repository.applied == []
    assert len(repository.completed) == 1


def test_worker_passes_retry_limit_for_dead_letter_transition() -> None:
    repository = FakeRepository()
    repository.events[0] = repository.events[0].model_copy(update={"attempts": 4})
    worker = MemoryExtractionWorker(
        repository,
        SimpleNamespace(build=lambda **kwargs: (_ for _ in ()).throw(ValueError("bad"))),
        SimpleNamespace(extract=lambda **kwargs: []),
        max_attempts=5,
    )

    assert worker.run_once(now=NOW) == 0
    assert repository.retried[0][1]["max_attempts"] == 5
    assert repository.retried[0][1]["available_at"] > NOW


def test_worker_retries_failed_extraction_without_writing() -> None:
    repository = FakeRepository()
    summary_service = SimpleNamespace(build=lambda **kwargs: (_ for _ in ()).throw(ValueError("bad summary")))
    extraction_service = SimpleNamespace(extract=lambda **kwargs: [])
    worker = MemoryExtractionWorker(repository, summary_service, extraction_service, worker_id="worker-1")

    assert worker.run_once(now=NOW) == 0
    assert repository.applied == []
    assert len(repository.retried) == 1
