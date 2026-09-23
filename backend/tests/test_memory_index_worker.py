from datetime import datetime, timedelta, timezone

from application.customer_service.memory_index_worker import MemoryIndexWorker
from domain.customer_service_agent.memory.models import (
    MemoryItem,
    MemoryOutboxEvent,
    MemoryOutboxEventType,
    MemoryStatus,
    MemoryType,
)


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)
MEMORY_ID = "550e8400-e29b-41d4-a716-446655440000"


class FakeRepository:
    def __init__(self, event_type=MemoryOutboxEventType.INDEX_UPSERT) -> None:
        self.events = [MemoryOutboxEvent(
            event_type=event_type,
            aggregate_id=MEMORY_ID,
            payload={"memory_id": MEMORY_ID, "user_id": "user-1"},
            available_at=NOW,
            created_at=NOW,
        )]
        self.completed = []

    def claim_outbox(self, *args, **kwargs):
        events, self.events = self.events, []
        return events

    def get_items_by_ids(self, user_id, memory_ids, now):
        return [MemoryItem(
            memory_id=MEMORY_ID,
            user_id="user-1",
            memory_type=MemoryType.FACT,
            memory_key="fact.goal",
            content="用户计划本月开户",
            confidence=0.9,
            status=MemoryStatus.ACTIVE,
            version=1,
            expires_at=NOW + timedelta(days=30),
            created_at=NOW,
            updated_at=NOW,
        )]

    def complete_outbox(self, event_id, processed_at):
        self.completed.append(str(event_id))

    def retry_outbox(self, *args, **kwargs):
        raise AssertionError("retry should not be called")


class FakeIndex:
    def __init__(self) -> None:
        self.upserts = []
        self.deletes = []

    def upsert(self, records):
        self.upserts.extend(records)

    def delete(self, memory_ids):
        self.deletes.extend(memory_ids)


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_index_worker_hydrates_mysql_before_upsert() -> None:
    repository = FakeRepository()
    index = FakeIndex()
    worker = MemoryIndexWorker(repository, index, FakeEmbedder(), worker_id="index-1")

    assert worker.run_once(now=NOW) == 1
    assert len(index.upserts) == 1
    assert index.upserts[0].user_id == "user-1"
    assert len(repository.completed) == 1


def test_upsert_event_deletes_stale_index_when_mysql_item_is_missing() -> None:
    class MissingRepository(FakeRepository):
        def get_items_by_ids(self, user_id, memory_ids, now):
            return []

    repository = MissingRepository()
    index = FakeIndex()
    worker = MemoryIndexWorker(repository, index, FakeEmbedder(), worker_id="index-1")

    assert worker.run_once(now=NOW) == 1
    assert index.upserts == []
    assert index.deletes == [MEMORY_ID]


def test_index_worker_passes_retry_limit_after_failure() -> None:
    class RetryRepository(FakeRepository):
        def __init__(self):
            super().__init__()
            self.events[0] = self.events[0].model_copy(update={"attempts": 4})
            self.retried = []

        def retry_outbox(self, event_id, **kwargs):
            self.retried.append((event_id, kwargs))

    class FailingIndex(FakeIndex):
        def upsert(self, records):
            raise RuntimeError("qdrant unavailable")

    repository = RetryRepository()
    worker = MemoryIndexWorker(
        repository,
        FailingIndex(),
        FakeEmbedder(),
        worker_id="index-1",
        max_attempts=5,
    )

    assert worker.run_once(now=NOW) == 0
    assert repository.retried[0][1]["max_attempts"] == 5
    assert repository.retried[0][1]["available_at"] > NOW


def test_index_worker_deletes_idempotently_without_mysql_content() -> None:
    repository = FakeRepository(MemoryOutboxEventType.INDEX_DELETE)
    index = FakeIndex()
    worker = MemoryIndexWorker(repository, index, FakeEmbedder(), worker_id="index-1")

    assert worker.run_once(now=NOW) == 1
    assert index.deletes == [MEMORY_ID]
