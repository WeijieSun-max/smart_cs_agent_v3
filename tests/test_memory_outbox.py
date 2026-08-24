from datetime import datetime, timezone

from infra.memory.mysql_conversation_archive import MySQLConversationArchive
from infra.memory.mysql_memory_repository import MySQLMemoryRepository


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc).isoformat()


class Cursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.outbox_keys: set[tuple[str, str]] = set()

    def execute(self, sql, args=None):
        self.calls.append((sql, args))
        if "INSERT IGNORE INTO cs_memory_outbox" in sql:
            key = ("turn_completed", str(args[1]))
            if key in self.outbox_keys:
                return 0
            self.outbox_keys.add(key)
        return 1


class TransactionClient:
    def __init__(self) -> None:
        self.cursor = Cursor()
        self.transaction_count = 0

    def execute_in_transaction(self, operation):
        self.transaction_count += 1
        return True, operation(self.cursor)


def _archive(client: TransactionClient) -> MySQLConversationArchive:
    archive = MySQLConversationArchive.__new__(MySQLConversationArchive)
    archive.mysql_client = client
    archive.user_id = "user-1"
    archive._ready = True
    return archive


def test_complete_turn_persists_assistant_and_outbox_in_one_transaction() -> None:
    client = TransactionClient()

    stored = _archive(client).complete_turn(
        "session-1",
        "处理完成",
        NOW,
        "turn-1",
        enqueue_memory=True,
    )

    sql = "\n".join(statement for statement, _ in client.cursor.calls)
    assert stored is True
    assert client.transaction_count == 1
    assert "INSERT IGNORE INTO cs_messages" in sql
    assert "INSERT IGNORE INTO cs_memory_outbox" in sql
    assert "turn_completed" in sql
    assert any("turn-1" in repr(args) for _, args in client.cursor.calls)


def test_duplicate_turn_keeps_one_outbox_aggregate() -> None:
    client = TransactionClient()
    archive = _archive(client)

    archive.complete_turn("session-1", "处理完成", NOW, "turn-1", enqueue_memory=True)
    archive.complete_turn("session-1", "处理完成", NOW, "turn-1", enqueue_memory=True)

    assert client.transaction_count == 2
    assert client.cursor.outbox_keys == {("turn_completed", "turn-1")}


def test_memory_outbox_schema_enforces_event_aggregate_uniqueness() -> None:
    class SchemaClient:
        def __init__(self) -> None:
            self.statements = []

        def execute_update(self, statement):
            self.statements.append(statement)
            return True, 0

    client = SchemaClient()
    repository = MySQLMemoryRepository(client)
    schema = "\n".join(client.statements)

    assert repository.available is True
    assert "UNIQUE KEY uk_cs_memory_outbox_event (event_type, aggregate_id)" in schema


def test_complete_turn_skips_outbox_when_layered_memory_is_disabled() -> None:
    client = TransactionClient()

    _archive(client).complete_turn(
        "session-1",
        "处理完成",
        NOW,
        "turn-1",
        enqueue_memory=False,
    )

    sql = "\n".join(statement for statement, _ in client.cursor.calls)
    assert "INSERT IGNORE INTO cs_messages" in sql
    assert "cs_memory_outbox" not in sql
