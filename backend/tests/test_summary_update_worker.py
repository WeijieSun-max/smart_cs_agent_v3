from datetime import datetime, timezone
from types import SimpleNamespace

from application.customer_service.summary_update_worker import SummaryUpdateWorker
from domain.customer_service_agent.memory.models import (
    MemoryOutboxEvent,
    MemoryOutboxEventType,
    SessionSummary,
)


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


class FakeRepository:
    def __init__(self, messages: list[dict], previous: SessionSummary | None = None) -> None:
        self.events = [MemoryOutboxEvent(
            event_type=MemoryOutboxEventType.SUMMARY_UPDATE,
            aggregate_id="turn-1",
            payload={"user_id": "user-1", "session_id": "session-1", "turn_id": "turn-1"},
            available_at=NOW,
            created_at=NOW,
        )]
        self.messages = messages
        self.previous = previous
        self.saved = []
        self.completed = []
        self.retried = []

    def claim_outbox(self, event_types, **kwargs):
        assert event_types == [MemoryOutboxEventType.SUMMARY_UPDATE]
        events, self.events = self.events, []
        return events

    def get_latest_summary(self, user_id, session_id):
        return self.previous

    def get_messages_after(self, user_id, session_id, after_message_id, limit=1000):
        return [item for item in self.messages if item["message_id"] > after_message_id]

    def save_summary(self, summary):
        self.saved.append(summary)

    def complete_outbox(self, event_id, processed_at):
        self.completed.append(str(event_id))

    def retry_outbox(self, event_id, **kwargs):
        self.retried.append((str(event_id), kwargs))


def _messages(turns: int, *, start: int = 1, content: str = "update") -> list[dict]:
    result = []
    message_id = start
    for index in range(turns):
        result.extend([
            {"message_id": message_id, "role": "user", "content": content, "turn_id": f"turn-{index}"},
            {"message_id": message_id + 1, "role": "assistant", "content": "ack", "turn_id": f"turn-{index}"},
        ])
        message_id += 2
    return result


def _summary(version: int, covers_until: int, text: str = "摘要") -> SessionSummary:
    return SessionSummary(
        user_id="user-1",
        session_id="session-1",
        version=version,
        summary_text=text,
        structured_data={},
        covers_until_message_id=covers_until,
        created_at=NOW,
        updated_at=NOW,
    )


def test_summary_worker_waits_for_initial_threshold() -> None:
    repository = FakeRepository(_messages(1))
    worker = SummaryUpdateWorker(
        repository,
        SimpleNamespace(build=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("summary must not run")
        )),
        summary_eligible_turns=2,
    )

    assert worker.run_once(now=NOW) == 1
    assert repository.saved == []
    assert len(repository.completed) == 1


def test_summary_worker_saves_and_caches_at_initial_threshold() -> None:
    repository = FakeRepository(_messages(2))
    summary = _summary(1, 4)
    cached = []
    worker = SummaryUpdateWorker(
        repository,
        SimpleNamespace(build=lambda **kwargs: summary),
        summary_cache=SimpleNamespace(
            cache_session_summary=lambda session_id, value: cached.append((session_id, value))
        ),
        summary_eligible_turns=2,
    )

    assert worker.run_once(now=NOW) == 1
    assert repository.saved == [summary]
    assert cached[0][0] == "session-1"
    assert cached[0][1]["version"] == 1


def test_summary_worker_waits_for_increment_threshold() -> None:
    previous = _summary(1, 2, "existing")
    repository = FakeRepository(_messages(1, start=3), previous)
    worker = SummaryUpdateWorker(
        repository,
        SimpleNamespace(build=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("summary must not run")
        )),
        summary_increment_turns=8,
    )

    assert worker.run_once(now=NOW) == 1
    assert repository.saved == []


def test_summary_worker_updates_at_increment_threshold() -> None:
    previous = _summary(1, 2, "existing")
    repository = FakeRepository(_messages(8, start=3), previous)
    next_summary = _summary(2, 18, "updated")
    captured = {}
    worker = SummaryUpdateWorker(
        repository,
        SimpleNamespace(build=lambda **kwargs: captured.update(kwargs) or next_summary),
        summary_increment_turns=8,
    )

    assert worker.run_once(now=NOW) == 1
    assert repository.saved == [next_summary]
    assert captured["previous"] == previous
    assert len(captured["new_messages"]) == 16


def test_summary_worker_allows_character_trigger() -> None:
    previous = _summary(1, 2, "existing")
    repository = FakeRepository(_messages(1, start=3, content="long enough update"), previous)
    next_summary = _summary(2, 4, "updated")
    worker = SummaryUpdateWorker(
        repository,
        SimpleNamespace(build=lambda **kwargs: next_summary),
        summary_increment_turns=8,
        summary_increment_chars=5,
    )

    assert worker.run_once(now=NOW) == 1
    assert repository.saved == [next_summary]


def test_summary_worker_retries_failure_independently() -> None:
    repository = FakeRepository(_messages(2))
    worker = SummaryUpdateWorker(
        repository,
        SimpleNamespace(build=lambda **kwargs: (_ for _ in ()).throw(ValueError("bad summary"))),
        summary_eligible_turns=2,
    )

    assert worker.run_once(now=NOW) == 0
    assert repository.saved == []
    assert len(repository.retried) == 1
