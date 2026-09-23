from __future__ import annotations

from datetime import datetime, timezone

from infra.memory.mysql_conversation_archive import MySQLConversationArchive
from infra.memory.mysql_memory_repository import MySQLMemoryRepository


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


class CascadeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.current_rows = []
        self.remaining_sources = {"single-memory": 0, "shared-memory": 1}

    def execute(self, sql, args=None):
        self.calls.append((sql, args))
        normalized = " ".join(sql.split())
        if normalized.startswith("SELECT DISTINCT s.memory_id"):
            self.current_rows = [
                {"memory_id": "single-memory", "content": "只来自当前会话"},
                {"memory_id": "shared-memory", "content": "来自多个会话"},
            ]
        elif normalized.startswith("SELECT COUNT(*)"):
            self.current_rows = [{"count": self.remaining_sources[str(args[0])]}]
        else:
            self.current_rows = []
        return 1

    def fetchall(self):
        return list(self.current_rows)

    def fetchone(self):
        return self.current_rows[0] if self.current_rows else None


class CascadeClient:
    def __init__(self, cursor: CascadeCursor) -> None:
        self.cursor = cursor

    def execute_in_transaction(self, operation):
        return True, operation(self.cursor)


def test_session_source_deletion_removes_single_source_and_keeps_shared_memory() -> None:
    cursor = CascadeCursor()
    repository = MySQLMemoryRepository.__new__(MySQLMemoryRepository)
    repository.mysql_client = CascadeClient(cursor)
    repository._ready = True

    deleted = repository.delete_session_sources("user-1", "session-1")

    assert deleted == ["single-memory"]
    item_deletes = [
        args for sql, args in cursor.calls if "DELETE FROM cs_memory_items" in sql
    ]
    assert item_deletes == [("single-memory", "user-1")]
    assert any("DELETE FROM cs_session_summaries" in sql for sql, _ in cursor.calls)
    assert any("DELETE FROM cs_memory_extraction_checkpoints" in sql for sql, _ in cursor.calls)
    assert any("INSERT INTO cs_memory_audit" in sql for sql, _ in cursor.calls)
    assert any("index_delete" in repr(args) for sql, args in cursor.calls if "cs_memory_outbox" in sql)


class ArchiveCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.current = None

    def execute(self, sql, args=None):
        self.calls.append((sql, args))
        self.current = {"session_id": "session-1"} if "SELECT session_id" in sql else None
        return 1

    def fetchone(self):
        return self.current


class ArchiveClient:
    def __init__(self) -> None:
        self.cursor = ArchiveCursor()

    def execute_in_transaction(self, operation):
        return True, operation(self.cursor)


class MemoryCascadeSpy:
    def __init__(self) -> None:
        self.calls = []

    def delete_session_sources_in_transaction(self, cursor, user_id, session_id):
        self.calls.append((cursor, user_id, session_id))
        return ["single-memory"]


def test_archive_deletes_session_and_memory_sources_in_one_transaction() -> None:
    client = ArchiveClient()
    cascade = MemoryCascadeSpy()
    archive = MySQLConversationArchive.__new__(MySQLConversationArchive)
    archive.mysql_client = client
    archive._default_user_id = "user-1"
    archive.memory_repository = cascade
    archive._ready = True

    removed = archive.delete_session("session-1")

    assert removed is True
    assert cascade.calls == [(client.cursor, "user-1", "session-1")]
    assert any("DELETE FROM cs_sessions" in sql for sql, _ in client.cursor.calls)
