from datetime import datetime, timezone

from infra.memory.mysql_memory_repository import MySQLMemoryRepository


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


class Cursor:
    def __init__(self, current_checkpoint: int) -> None:
        self.current_checkpoint = current_checkpoint
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql, args=None):
        self.calls.append((sql, args))
        return 1

    def fetchone(self):
        return {"last_message_id": self.current_checkpoint}


class Client:
    def __init__(self, current_checkpoint: int) -> None:
        self.cursor = Cursor(current_checkpoint)

    def execute_in_transaction(self, operation):
        return True, operation(self.cursor)


def _repository(current_checkpoint: int) -> tuple[MySQLMemoryRepository, Cursor]:
    client = Client(current_checkpoint)
    repository = MySQLMemoryRepository.__new__(MySQLMemoryRepository)
    repository.mysql_client = client
    repository._ready = True
    return repository, client.cursor


def test_apply_memory_extraction_advances_checkpoint_atomically() -> None:
    repository, cursor = _repository(3)

    applied = repository.apply_memory_extraction(
        user_id="user-1",
        session_id="session-1",
        expected_last_message_id=3,
        last_message_id=5,
        items=[],
        sources=[],
        superseded_ids=[],
        updated_at=NOW,
    )

    assert applied is True
    assert any(
        "UPDATE cs_memory_extraction_checkpoints" in sql and args[0] == 5
        for sql, args in cursor.calls
    )


def test_apply_memory_extraction_rejects_stale_checkpoint_before_writes() -> None:
    repository, cursor = _repository(3)

    applied = repository.apply_memory_extraction(
        user_id="user-1",
        session_id="session-1",
        expected_last_message_id=1,
        last_message_id=5,
        items=[],
        sources=[],
        superseded_ids=[],
        updated_at=NOW,
    )

    assert applied is False
    assert not any("SET last_message_id" in sql for sql, _ in cursor.calls)


def test_apply_memory_extraction_treats_already_covered_range_as_success() -> None:
    repository, cursor = _repository(7)

    applied = repository.apply_memory_extraction(
        user_id="user-1",
        session_id="session-1",
        expected_last_message_id=3,
        last_message_id=5,
        items=[],
        sources=[],
        superseded_ids=[],
        updated_at=NOW,
    )

    assert applied is True
    assert not any("SET last_message_id" in sql for sql, _ in cursor.calls)
