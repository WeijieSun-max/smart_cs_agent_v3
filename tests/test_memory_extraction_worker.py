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
            event_type=MemoryOutboxEventType.MEMORY_EXTRACT,
            aggregate_id="turn-1",
            payload={"user_id": "user-1", "session_id": "session-1", "turn_id": "turn-1"},
            available_at=NOW,
            created_at=NOW,
        )]
        self.messages = [
            {"message_id": 1, "role": "user", "content": "我偏好稳健产品", "turn_id": "turn-1"},
            {"message_id": 2, "role": "assistant", "content": "已了解", "turn_id": "turn-1"},
        ]
        self.checkpoint = 0
        self.latest_summary = None
        self.completed: list[str] = []
        self.retried: list[tuple[str, dict]] = []
        self.applied: list[dict] = []
        self.apply_result = True

    def claim_outbox(self, event_types, **kwargs):
        assert event_types == [MemoryOutboxEventType.MEMORY_EXTRACT]
        events, self.events = self.events, []
        return events

    def get_extraction_checkpoint(self, user_id, session_id):
        return self.checkpoint

    def get_latest_summary(self, user_id, session_id):
        return self.latest_summary

    def get_messages_after(self, user_id, session_id, after_message_id, limit=1000):
        return [item for item in self.messages if item["message_id"] > after_message_id]

    def find_active_by_key(self, user_id, memory_type, memory_key):
        return None

    def apply_memory_extraction(self, **kwargs):
        self.applied.append(kwargs)
        if self.apply_result:
            self.checkpoint = kwargs["last_message_id"]
        return self.apply_result

    def complete_outbox(self, event_id, processed_at):
        self.completed.append(str(event_id))

    def retry_outbox(self, event_id, **kwargs):
        self.retried.append((str(event_id), kwargs))


def _candidate() -> MemoryCandidate:
    return MemoryCandidate(
        memory_type=MemoryType.PREFERENCE,
        memory_key="preference.risk",
        content="用户偏好稳健产品",
        confidence=0.9,
    )


def test_worker_extracts_memory_and_advances_independent_checkpoint() -> None:
    repository = FakeRepository()
    captured = {}

    def extract(**kwargs):
        captured.update(kwargs)
        return [_candidate()]

    worker = MemoryExtractionWorker(
        repository,
        SimpleNamespace(extract=extract),
        worker_id="worker-1",
    )

    assert worker.run_once(now=NOW) == 1
    assert len(repository.applied) == 1
    applied = repository.applied[0]
    assert applied["expected_last_message_id"] == 0
    assert applied["last_message_id"] == 2
    assert len(applied["items"]) == 1
    assert captured["summary_text"] == ""
    assert repository.checkpoint == 2
    assert len(repository.completed) == 1


def test_worker_can_use_latest_summary_without_waiting_for_summary_update() -> None:
    repository = FakeRepository()
    repository.latest_summary = SessionSummary(
        user_id="user-1",
        session_id="session-1",
        version=1,
        summary_text="已有摘要",
        structured_data={},
        covers_until_message_id=1,
        created_at=NOW,
        updated_at=NOW,
    )
    captured = {}
    worker = MemoryExtractionWorker(
        repository,
        SimpleNamespace(extract=lambda **kwargs: captured.update(kwargs) or []),
    )

    assert worker.run_once(now=NOW) == 1
    assert captured["summary_text"] == "已有摘要"
    assert repository.applied[0]["items"] == []
    assert repository.checkpoint == 2


def test_worker_waits_until_extraction_turn_threshold() -> None:
    repository = FakeRepository()
    worker = MemoryExtractionWorker(
        repository,
        SimpleNamespace(extract=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("extraction must not run")
        )),
        extraction_increment_turns=2,
    )

    assert worker.run_once(now=NOW) == 1
    assert repository.applied == []
    assert repository.checkpoint == 0
    assert len(repository.completed) == 1


def test_worker_retries_checkpoint_conflict_without_duplicate_write() -> None:
    repository = FakeRepository()
    repository.apply_result = False
    worker = MemoryExtractionWorker(
        repository,
        SimpleNamespace(extract=lambda **kwargs: []),
    )

    assert worker.run_once(now=NOW) == 0
    assert len(repository.applied) == 1
    assert repository.completed == []
    assert len(repository.retried) == 1


def test_worker_passes_retry_limit_for_dead_letter_transition() -> None:
    repository = FakeRepository()
    repository.events[0] = repository.events[0].model_copy(update={"attempts": 4})
    worker = MemoryExtractionWorker(
        repository,
        SimpleNamespace(extract=lambda **kwargs: (_ for _ in ()).throw(ValueError("bad"))),
        max_attempts=5,
    )

    assert worker.run_once(now=NOW) == 0
    assert repository.retried[0][1]["max_attempts"] == 5
    assert repository.retried[0][1]["available_at"] > NOW


def test_worker_retries_failed_extraction_without_writing() -> None:
    repository = FakeRepository()
    worker = MemoryExtractionWorker(
        repository,
        SimpleNamespace(extract=lambda **kwargs: (_ for _ in ()).throw(ValueError("bad extraction"))),
        worker_id="worker-1",
    )

    assert worker.run_once(now=NOW) == 0
    assert repository.applied == []
    assert len(repository.retried) == 1
