from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from domain.customer_service_agent.interfaces.i_memory_repository import IMemoryRepository
from domain.customer_service_agent.memory.models import (
    MemoryItem,
    MemoryOutboxEvent,
    MemoryOutboxEventType,
    MemorySource,
    MemoryStatus,
    MemoryType,
    SessionSummary,
)
from infra.db.mysql_client import MySQLClient
from pkg.exceptions.exception import StorageOperationError, StorageUnavailableError


class MySQLMemoryRepository(IMemoryRepository):
    def __init__(self, mysql_client: MySQLClient | None) -> None:
        self.mysql_client = mysql_client
        self._ready = bool(mysql_client is not None and self._ensure_tables())

    @property
    def available(self) -> bool:
        return self.mysql_client is not None and self._ready

    def _ensure_tables(self) -> bool:
        assert self.mysql_client is not None
        statements = [
            """
            CREATE TABLE IF NOT EXISTS cs_session_summaries (
                summary_id CHAR(36) PRIMARY KEY,
                user_id VARCHAR(128) NOT NULL,
                session_id VARCHAR(128) NOT NULL,
                version INT NOT NULL,
                summary_text LONGTEXT NOT NULL,
                structured_json LONGTEXT NOT NULL,
                covers_until_message_id BIGINT UNSIGNED NOT NULL,
                created_at DATETIME(6) NOT NULL,
                updated_at DATETIME(6) NOT NULL,
                UNIQUE KEY uk_cs_summary_user_session_version (user_id, session_id, version),
                INDEX idx_cs_summary_latest (user_id, session_id, version)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS cs_memory_items (
                memory_id CHAR(36) PRIMARY KEY,
                user_id VARCHAR(128) NOT NULL,
                memory_type VARCHAR(32) NOT NULL,
                memory_key VARCHAR(255) NOT NULL,
                content LONGTEXT NOT NULL,
                structured_json LONGTEXT NOT NULL,
                confidence DOUBLE NOT NULL,
                status VARCHAR(32) NOT NULL,
                version INT NOT NULL,
                supersedes_id CHAR(36) NULL,
                valid_from DATETIME(6) NULL,
                valid_until DATETIME(6) NULL,
                expires_at DATETIME(6) NULL,
                created_at DATETIME(6) NOT NULL,
                updated_at DATETIME(6) NOT NULL,
                INDEX idx_cs_memory_active (user_id, status, memory_type, expires_at),
                INDEX idx_cs_memory_key (user_id, memory_type, memory_key, status),
                INDEX idx_cs_memory_updated (user_id, updated_at, memory_id)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS cs_memory_sources (
                source_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                memory_id CHAR(36) NOT NULL,
                session_id VARCHAR(128) NOT NULL,
                turn_id VARCHAR(64) NOT NULL,
                message_id BIGINT UNSIGNED NULL,
                source_kind VARCHAR(32) NOT NULL,
                created_at DATETIME(6) NOT NULL,
                UNIQUE KEY uk_cs_memory_source (memory_id, session_id, turn_id, source_kind),
                INDEX idx_cs_memory_source_session (session_id, memory_id),
                CONSTRAINT fk_cs_memory_source_item FOREIGN KEY (memory_id)
                    REFERENCES cs_memory_items(memory_id) ON DELETE CASCADE
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS cs_memory_outbox (
                event_id CHAR(36) PRIMARY KEY,
                event_type VARCHAR(32) NOT NULL,
                aggregate_id VARCHAR(128) NOT NULL,
                payload_json LONGTEXT NOT NULL,
                status VARCHAR(32) NOT NULL,
                attempts INT NOT NULL DEFAULT 0,
                available_at DATETIME(6) NOT NULL,
                lease_until DATETIME(6) NULL,
                worker_id VARCHAR(128) NULL,
                last_error_code VARCHAR(128) NULL,
                created_at DATETIME(6) NOT NULL,
                processed_at DATETIME(6) NULL,
                UNIQUE KEY uk_cs_memory_outbox_event (event_type, aggregate_id),
                INDEX idx_cs_memory_outbox_claim (status, event_type, available_at, lease_until)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS cs_memory_audit (
                audit_id CHAR(36) PRIMARY KEY,
                memory_id CHAR(36) NOT NULL,
                user_id VARCHAR(128) NOT NULL,
                action VARCHAR(32) NOT NULL,
                actor VARCHAR(64) NOT NULL,
                reason VARCHAR(255) NOT NULL,
                before_json LONGTEXT NULL,
                after_json LONGTEXT NULL,
                content_hash CHAR(64) NULL,
                created_at DATETIME(6) NOT NULL,
                INDEX idx_cs_memory_audit_user (user_id, created_at),
                INDEX idx_cs_memory_audit_item (memory_id, created_at)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
        ]
        return all(self.mysql_client.execute_update(statement)[0] for statement in statements)

    def _client(self) -> MySQLClient:
        if not self.available or self.mysql_client is None:
            raise StorageUnavailableError()
        return self.mysql_client

    def get_latest_summary(self, user_id: str, session_id: str) -> SessionSummary | None:
        ok, row = self._client().execute_query(
            """
            SELECT summary_id, user_id, session_id, version, summary_text, structured_json,
                   covers_until_message_id, created_at, updated_at
            FROM cs_session_summaries
            WHERE user_id = %s AND session_id = %s
            ORDER BY version DESC LIMIT 1
            """,
            (user_id, session_id),
            fetch_one=True,
        )
        if not ok:
            raise StorageOperationError()
        return _summary_from_row(row) if row else None

    def save_summary(self, summary: SessionSummary) -> None:
        ok, _ = self._client().execute_update(
            """
            INSERT INTO cs_session_summaries
                (summary_id, user_id, session_id, version, summary_text, structured_json,
                 covers_until_message_id, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(summary.summary_id), summary.user_id, summary.session_id, summary.version,
                summary.summary_text, _json(summary.structured_data), summary.covers_until_message_id,
                _db_time(summary.created_at), _db_time(summary.updated_at),
            ),
        )
        if not ok:
            raise StorageOperationError()

    def get_messages_after(
        self,
        user_id: str,
        session_id: str,
        after_message_id: int,
        limit: int = 1000,
    ) -> list[dict[str, object]]:
        ok, rows = self._client().execute_query(
            """
            SELECT m.message_id, m.role, m.content, m.turn_id, m.created_at
            FROM cs_messages m JOIN cs_sessions s ON s.session_id = m.session_id
            WHERE s.user_id = %s AND m.session_id = %s AND m.message_id > %s
            ORDER BY m.message_id LIMIT %s
            """,
            (user_id, session_id, after_message_id, limit),
        )
        if not ok:
            raise StorageOperationError()
        return [
            {
                "message_id": int(row["message_id"]),
                "role": str(row["role"]),
                "content": str(row["content"]),
                "turn_id": str(row["turn_id"]) if row.get("turn_id") else "",
                "timestamp": _aware(row["created_at"]),
            }
            for row in rows
        ]

    def get_items_by_ids(self, user_id: str, memory_ids: list[str], now: datetime) -> list[MemoryItem]:
        if not memory_ids:
            return []
        placeholders = ", ".join(["%s"] * len(memory_ids))
        ok, rows = self._client().execute_query(
            f"""
            SELECT * FROM cs_memory_items
            WHERE user_id = %s AND memory_id IN ({placeholders})
              AND status = 'active' AND (expires_at IS NULL OR expires_at > %s)
            """,
            (user_id, *memory_ids, _db_time(now)),
        )
        if not ok:
            raise StorageOperationError()
        return [_item_from_row(row) for row in rows]

    def list_active_items(
        self,
        user_id: str,
        *,
        memory_types: list[MemoryType] | None = None,
        limit: int = 20,
        now: datetime,
    ) -> list[MemoryItem]:
        clauses = ["user_id = %s", "status = 'active'", "(expires_at IS NULL OR expires_at > %s)"]
        args: list[Any] = [user_id, _db_time(now)]
        if memory_types:
            clauses.append(f"memory_type IN ({', '.join(['%s'] * len(memory_types))})")
            args.extend(item.value for item in memory_types)
        args.append(limit)
        ok, rows = self._client().execute_query(
            f"SELECT * FROM cs_memory_items WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT %s",
            tuple(args),
        )
        if not ok:
            raise StorageOperationError()
        return [_item_from_row(row) for row in rows]

    def scan_active_items(
        self,
        *,
        cursor: str | None,
        limit: int,
        now: datetime,
    ) -> tuple[list[MemoryItem], str | None]:
        clauses = ["status = 'active'", "(expires_at IS NULL OR expires_at > %s)"]
        args: list[Any] = [_db_time(now)]
        if cursor:
            clauses.append("memory_id > %s")
            args.append(cursor)
        args.append(limit + 1)
        ok, rows = self._client().execute_query(
            f"SELECT * FROM cs_memory_items WHERE {' AND '.join(clauses)} ORDER BY memory_id LIMIT %s",
            tuple(args),
        )
        if not ok:
            raise StorageOperationError()
        items = [_item_from_row(row) for row in rows[:limit]]
        next_cursor = str(items[-1].memory_id) if len(rows) > limit and items else None
        return items, next_cursor

    def find_active_by_key(self, user_id: str, memory_type: MemoryType, memory_key: str) -> MemoryItem | None:
        ok, row = self._client().execute_query(
            """
            SELECT * FROM cs_memory_items
            WHERE user_id = %s AND memory_type = %s AND memory_key = %s AND status = 'active'
            ORDER BY version DESC LIMIT 1
            """,
            (user_id, memory_type.value, memory_key),
            fetch_one=True,
        )
        if not ok:
            raise StorageOperationError()
        return _item_from_row(row) if row else None

    def save_item(self, item: MemoryItem, sources: list[MemorySource]) -> None:
        def operation(cursor) -> None:
            cursor.execute(
                """
                INSERT INTO cs_memory_items
                    (memory_id, user_id, memory_type, memory_key, content, structured_json, confidence,
                     status, version, supersedes_id, valid_from, valid_until, expires_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                _item_args(item),
            )
            for source in sources:
                cursor.execute(
                    """
                    INSERT IGNORE INTO cs_memory_sources
                        (memory_id, session_id, turn_id, message_id, source_kind, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(source.memory_id), source.session_id, source.turn_id, source.message_id,
                        source.source_kind, _db_time(source.created_at),
                    ),
                )

        ok, _ = self._client().execute_in_transaction(operation)
        if not ok:
            raise StorageOperationError()

    def apply_extraction(
        self,
        summary: SessionSummary,
        items: list[MemoryItem],
        sources: list[MemorySource],
        superseded_ids: list[str],
    ) -> None:
        now = summary.updated_at
        source_map: dict[str, list[MemorySource]] = {}
        for source in sources:
            source_map.setdefault(str(source.memory_id), []).append(source)

        def operation(cursor) -> None:
            cursor.execute(
                """
                INSERT INTO cs_session_summaries
                    (summary_id, user_id, session_id, version, summary_text, structured_json,
                     covers_until_message_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(summary.summary_id), summary.user_id, summary.session_id, summary.version,
                    summary.summary_text, _json(summary.structured_data), summary.covers_until_message_id,
                    _db_time(summary.created_at), _db_time(summary.updated_at),
                ),
            )
            for memory_id in superseded_ids:
                cursor.execute(
                    "UPDATE cs_memory_items SET status = 'superseded', updated_at = %s WHERE memory_id = %s AND user_id = %s",
                    (_db_time(now), memory_id, summary.user_id),
                )
                self._insert_index_event(
                    cursor, MemoryOutboxEventType.INDEX_DELETE, memory_id, now, user_id=summary.user_id
                )
            for item in items:
                cursor.execute(
                    """
                    INSERT INTO cs_memory_items
                        (memory_id, user_id, memory_type, memory_key, content, structured_json, confidence,
                         status, version, supersedes_id, valid_from, valid_until, expires_at, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    _item_args(item),
                )
                cursor.execute(
                    """
                    INSERT INTO cs_memory_audit
                        (audit_id, memory_id, user_id, action, actor, reason, before_json, after_json, content_hash, created_at)
                    VALUES (%s, %s, %s, 'activated', 'system', 'conversation_extraction', NULL, %s, NULL, %s)
                    """,
                    (
                        str(uuid.uuid4()), str(item.memory_id), item.user_id,
                        _json({"type": item.memory_type.value, "version": item.version}), _db_time(now),
                    ),
                )
                self._insert_index_event(
                    cursor, MemoryOutboxEventType.INDEX_UPSERT, str(item.memory_id), now, user_id=item.user_id
                )
            for memory_id, memory_sources in source_map.items():
                for source in memory_sources:
                    cursor.execute(
                        """
                        INSERT IGNORE INTO cs_memory_sources
                            (memory_id, session_id, turn_id, message_id, source_kind, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            memory_id, source.session_id, source.turn_id, source.message_id,
                            source.source_kind, _db_time(source.created_at),
                        ),
                    )

        ok, _ = self._client().execute_in_transaction(operation)
        if not ok:
            raise StorageOperationError()

    @staticmethod
    def _insert_index_event(
        cursor,
        event_type: MemoryOutboxEventType,
        memory_id: str,
        now: datetime,
        *,
        user_id: str,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO cs_memory_outbox
                (event_id, event_type, aggregate_id, payload_json, status, attempts, available_at, created_at)
            VALUES (%s, %s, %s, %s, 'pending', 0, %s, %s)
            ON DUPLICATE KEY UPDATE status = 'pending', available_at = VALUES(available_at), processed_at = NULL
            """,
            (
                str(uuid.uuid4()), event_type.value, memory_id,
                _json({"memory_id": memory_id, "user_id": user_id}),
                _db_time(now), _db_time(now),
            ),
        )

    def transition_item(self, user_id: str, memory_id: str, status: MemoryStatus, reason: str) -> bool:
        now = _utc_now()
        statements = [
            (
                "UPDATE cs_memory_items SET status = %s, updated_at = %s WHERE memory_id = %s AND user_id = %s",
                (status.value, _db_time(now), memory_id, user_id),
            ),
            (
                """
                INSERT INTO cs_memory_audit
                    (audit_id, memory_id, user_id, action, actor, reason, before_json, after_json, content_hash, created_at)
                VALUES (%s, %s, %s, %s, 'system', %s, NULL, %s, NULL, %s)
                """,
                (str(uuid.uuid4()), memory_id, user_id, "transition", reason, _json({"status": status.value}), _db_time(now)),
            ),
        ]
        ok, affected = self._client().execute_transaction(statements)
        if not ok:
            raise StorageOperationError()
        return bool(affected)

    def list_items(
        self,
        user_id: str,
        *,
        memory_type: MemoryType | None = None,
        status: MemoryStatus | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[MemoryItem], str | None]:
        clauses = ["user_id = %s"]
        args: list[Any] = [user_id]
        if memory_type is not None:
            clauses.append("memory_type = %s")
            args.append(memory_type.value)
        if status is not None:
            clauses.append("status = %s")
            args.append(status.value)
        if cursor:
            clauses.append("memory_id > %s")
            args.append(cursor)
        args.append(limit + 1)
        ok, rows = self._client().execute_query(
            f"SELECT * FROM cs_memory_items WHERE {' AND '.join(clauses)} ORDER BY memory_id LIMIT %s",
            tuple(args),
        )
        if not ok:
            raise StorageOperationError()
        items = [_item_from_row(row) for row in rows[:limit]]
        next_cursor = str(items[-1].memory_id) if len(rows) > limit and items else None
        return items, next_cursor

    def correct_item(
        self,
        user_id: str,
        memory_id: str,
        *,
        content: str,
        structured_data: dict[str, Any] | None,
        reason: str,
        actor: str,
    ) -> MemoryItem | None:
        now = _utc_now()

        def operation(cursor) -> MemoryItem | None:
            cursor.execute(
                "SELECT * FROM cs_memory_items WHERE memory_id = %s AND user_id = %s AND status = 'active' FOR UPDATE",
                (memory_id, user_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            previous = _item_from_row(row)
            corrected = MemoryItem(
                user_id=user_id,
                memory_type=previous.memory_type,
                memory_key=previous.memory_key,
                content=content,
                structured_data=previous.structured_data if structured_data is None else structured_data,
                confidence=1.0,
                status=MemoryStatus.ACTIVE,
                version=previous.version + 1,
                supersedes_id=previous.memory_id,
                valid_from=now,
                valid_until=previous.valid_until,
                expires_at=previous.expires_at,
                created_at=now,
                updated_at=now,
            )
            cursor.execute(
                "UPDATE cs_memory_items SET status = 'superseded', updated_at = %s WHERE memory_id = %s AND user_id = %s",
                (_db_time(now), memory_id, user_id),
            )
            cursor.execute(
                """
                INSERT INTO cs_memory_items
                    (memory_id, user_id, memory_type, memory_key, content, structured_json, confidence,
                     status, version, supersedes_id, valid_from, valid_until, expires_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                _item_args(corrected),
            )
            cursor.execute(
                """
                INSERT INTO cs_memory_sources
                    (memory_id, session_id, turn_id, message_id, source_kind, created_at)
                VALUES (%s, '__memory_management__', %s, NULL, 'user_correction', %s)
                """,
                (str(corrected.memory_id), f"correction-{str(corrected.memory_id)[:24]}", _db_time(now)),
            )
            cursor.execute(
                """
                INSERT INTO cs_memory_audit
                    (audit_id, memory_id, user_id, action, actor, reason, before_json, after_json, content_hash, created_at)
                VALUES (%s, %s, %s, 'corrected', %s, %s, %s, %s, NULL, %s)
                """,
                (
                    str(uuid.uuid4()), str(corrected.memory_id), user_id, actor, reason,
                    _json({"memory_id": memory_id, "version": previous.version}),
                    _json({"memory_id": str(corrected.memory_id), "version": corrected.version}),
                    _db_time(now),
                ),
            )
            self._insert_index_event(
                cursor, MemoryOutboxEventType.INDEX_DELETE, memory_id, now, user_id=user_id
            )
            self._insert_index_event(
                cursor,
                MemoryOutboxEventType.INDEX_UPSERT,
                str(corrected.memory_id),
                now,
                user_id=user_id,
            )
            return corrected

        ok, corrected = self._client().execute_in_transaction(operation)
        if not ok:
            raise StorageOperationError()
        return corrected

    def hard_delete_item(self, user_id: str, memory_id: str, *, reason: str, actor: str) -> bool:
        now = _utc_now()

        def operation(cursor) -> bool:
            cursor.execute(
                "SELECT memory_id, content FROM cs_memory_items WHERE memory_id = %s AND user_id = %s FOR UPDATE",
                (memory_id, user_id),
            )
            row = cursor.fetchone()
            if not row:
                return False
            content_hash = hashlib.sha256(str(row["content"]).encode("utf-8")).hexdigest()
            cursor.execute(
                """
                INSERT INTO cs_memory_audit
                    (audit_id, memory_id, user_id, action, actor, reason, before_json, after_json, content_hash, created_at)
                VALUES (%s, %s, %s, 'forgotten', %s, %s, NULL, NULL, %s, %s)
                """,
                (str(uuid.uuid4()), memory_id, user_id, actor, reason, content_hash, _db_time(now)),
            )
            cursor.execute("DELETE FROM cs_memory_items WHERE memory_id = %s AND user_id = %s", (memory_id, user_id))
            cursor.execute(
                """
                INSERT INTO cs_memory_outbox
                    (event_id, event_type, aggregate_id, payload_json, status, attempts, available_at, created_at)
                VALUES (%s, 'index_delete', %s, %s, 'pending', 0, %s, %s)
                ON DUPLICATE KEY UPDATE status = 'pending', available_at = VALUES(available_at), processed_at = NULL
                """,
                (str(uuid.uuid4()), memory_id, _json({"memory_id": memory_id}), _db_time(now), _db_time(now)),
            )
            return True

        ok, removed = self._client().execute_in_transaction(operation)
        if not ok:
            raise StorageOperationError()
        return bool(removed)

    def purge_items(self, user_id: str, memory_type: MemoryType | None, *, reason: str, actor: str) -> int:
        clauses = ["user_id = %s"]
        args: list[Any] = [user_id]
        if memory_type is not None:
            clauses.append("memory_type = %s")
            args.append(memory_type.value)
        now = _utc_now()

        def operation(cursor) -> int:
            cursor.execute(
                f"SELECT memory_id, content FROM cs_memory_items WHERE {' AND '.join(clauses)} FOR UPDATE",
                tuple(args),
            )
            rows = cursor.fetchall()
            for row in rows:
                memory_id = str(row["memory_id"])
                content_hash = hashlib.sha256(str(row["content"]).encode("utf-8")).hexdigest()
                cursor.execute(
                    """
                    INSERT INTO cs_memory_audit
                        (audit_id, memory_id, user_id, action, actor, reason, before_json, after_json, content_hash, created_at)
                    VALUES (%s, %s, %s, 'forgotten', %s, %s, NULL, NULL, %s, %s)
                    """,
                    (str(uuid.uuid4()), memory_id, user_id, actor, reason, content_hash, _db_time(now)),
                )
                cursor.execute(
                    "DELETE FROM cs_memory_items WHERE memory_id = %s AND user_id = %s",
                    (memory_id, user_id),
                )
                self._insert_index_event(
                    cursor,
                    MemoryOutboxEventType.INDEX_DELETE,
                    memory_id,
                    now,
                    user_id=user_id,
                )
            return len(rows)

        ok, deleted = self._client().execute_in_transaction(operation)
        if not ok:
            raise StorageOperationError()
        return int(deleted)

    def delete_session_sources(self, user_id: str, session_id: str) -> list[str]:
        now = _utc_now()

        def operation(cursor) -> list[str]:
            return self.delete_session_sources_in_transaction(cursor, user_id, session_id, now=now)

        ok, deleted = self._client().execute_in_transaction(operation)
        if not ok:
            raise StorageOperationError()
        return deleted

    def delete_session_sources_in_transaction(
        self,
        cursor,
        user_id: str,
        session_id: str,
        *,
        now: datetime | None = None,
    ) -> list[str]:
        now = now or _utc_now()
        cursor.execute(
            """
            SELECT DISTINCT s.memory_id, i.content
            FROM cs_memory_sources s JOIN cs_memory_items i ON i.memory_id = s.memory_id
            WHERE s.session_id = %s AND i.user_id = %s FOR UPDATE
            """,
            (session_id, user_id),
        )
        rows = cursor.fetchall()
        content_by_id = {str(row["memory_id"]): str(row["content"]) for row in rows}
        cursor.execute(
            """
            DELETE s FROM cs_memory_sources s
            JOIN cs_memory_items i ON i.memory_id = s.memory_id
            WHERE s.session_id = %s AND i.user_id = %s
            """,
            (session_id, user_id),
        )
        deleted: list[str] = []
        for memory_id, content in content_by_id.items():
            cursor.execute("SELECT COUNT(*) AS count FROM cs_memory_sources WHERE memory_id = %s", (memory_id,))
            count_row = cursor.fetchone() or {"count": 0}
            if int(count_row["count"]):
                continue
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            cursor.execute(
                """
                INSERT INTO cs_memory_audit
                    (audit_id, memory_id, user_id, action, actor, reason, before_json, after_json, content_hash, created_at)
                VALUES (%s, %s, %s, 'forgotten', 'user', 'session_deleted', NULL, NULL, %s, %s)
                """,
                (str(uuid.uuid4()), memory_id, user_id, content_hash, _db_time(now)),
            )
            cursor.execute("DELETE FROM cs_memory_items WHERE memory_id = %s AND user_id = %s", (memory_id, user_id))
            self._insert_index_event(
                cursor,
                MemoryOutboxEventType.INDEX_DELETE,
                memory_id,
                now,
                user_id=user_id,
            )
            deleted.append(memory_id)
        cursor.execute(
            "DELETE FROM cs_session_summaries WHERE user_id = %s AND session_id = %s",
            (user_id, session_id),
        )
        return deleted

    def enqueue_outbox(self, event: MemoryOutboxEvent) -> bool:
        ok, affected = self._client().execute_update(
            """
            INSERT IGNORE INTO cs_memory_outbox
                (event_id, event_type, aggregate_id, payload_json, status, attempts, available_at,
                 lease_until, last_error_code, created_at, processed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(event.event_id), event.event_type.value, event.aggregate_id, _json(event.payload),
                event.status.value, event.attempts, _db_time(event.available_at), _db_time(event.lease_until),
                event.last_error_code, _db_time(event.created_at), _db_time(event.processed_at),
            ),
        )
        if not ok:
            raise StorageOperationError()
        return bool(affected)

    def claim_outbox(
        self,
        event_types: list[MemoryOutboxEventType],
        *,
        worker_id: str,
        limit: int,
        lease_seconds: int,
        now: datetime,
    ) -> list[MemoryOutboxEvent]:
        values = [item.value if isinstance(item, MemoryOutboxEventType) else str(item) for item in event_types]
        lease_until = now + timedelta(seconds=lease_seconds)

        def operation(cursor) -> list[MemoryOutboxEvent]:
            placeholders = ", ".join(["%s"] * len(values))
            cursor.execute(
                f"""
                SELECT * FROM cs_memory_outbox
                WHERE event_type IN ({placeholders})
                  AND status IN ('pending', 'retry', 'processing')
                  AND available_at <= %s
                  AND (lease_until IS NULL OR lease_until <= %s)
                ORDER BY created_at LIMIT %s FOR UPDATE SKIP LOCKED
                """,
                (*values, _db_time(now), _db_time(now), limit),
            )
            rows = cursor.fetchall()
            if not rows:
                return []
            ids = [str(row["event_id"]) for row in rows]
            cursor.execute(
                f"""
                UPDATE cs_memory_outbox
                SET status = 'processing', worker_id = %s, lease_until = %s
                WHERE event_id IN ({', '.join(['%s'] * len(ids))})
                """,
                (worker_id, _db_time(lease_until), *ids),
            )
            return [_outbox_from_row(row) for row in rows]

        ok, events = self._client().execute_in_transaction(operation)
        if not ok:
            raise StorageOperationError()
        return events

    def complete_outbox(self, event_id: str, processed_at: datetime) -> None:
        ok, _ = self._client().execute_update(
            """
            UPDATE cs_memory_outbox
            SET status = 'completed', processed_at = %s, lease_until = NULL, worker_id = NULL
            WHERE event_id = %s
            """,
            (_db_time(processed_at), event_id),
        )
        if not ok:
            raise StorageOperationError()

    def retry_outbox(
        self,
        event_id: str,
        *,
        error_code: str,
        available_at: datetime,
        max_attempts: int,
    ) -> None:
        ok, _ = self._client().execute_update(
            """
            UPDATE cs_memory_outbox
            SET attempts = attempts + 1,
                status = CASE WHEN attempts + 1 >= %s THEN 'dead' ELSE 'retry' END,
                available_at = %s, lease_until = NULL, worker_id = NULL, last_error_code = %s
            WHERE event_id = %s
            """,
            (max_attempts, _db_time(available_at), error_code[:128], event_id),
        )
        if not ok:
            raise StorageOperationError()

    def outbox_stats(self, now: datetime) -> dict[str, int | float | None]:
        ok, row = self._client().execute_query(
            """
            SELECT
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_count,
                SUM(CASE WHEN status = 'retry' THEN 1 ELSE 0 END) AS retry_count,
                SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) AS processing_count,
                SUM(CASE WHEN status = 'dead' THEN 1 ELSE 0 END) AS dead_count,
                MIN(CASE WHEN status IN ('pending', 'retry') THEN available_at ELSE NULL END) AS oldest_available_at
            FROM cs_memory_outbox
            """,
            fetch_one=True,
        )
        if not ok:
            raise StorageOperationError()
        row = row or {}
        oldest = _aware(row.get("oldest_available_at"))
        oldest_age = max(0.0, (now - oldest).total_seconds()) if oldest is not None else None
        return {
            "pending": int(row.get("pending_count") or 0),
            "retry": int(row.get("retry_count") or 0),
            "processing": int(row.get("processing_count") or 0),
            "dead": int(row.get("dead_count") or 0),
            "oldest_age_seconds": oldest_age,
        }

    def replay_dead_letter(self, event_id: str, now: datetime) -> bool:
        ok, affected = self._client().execute_update(
            """
            UPDATE cs_memory_outbox
            SET status = 'retry', attempts = 0, available_at = %s, lease_until = NULL,
                worker_id = NULL, last_error_code = NULL, processed_at = NULL
            WHERE event_id = %s AND status = 'dead'
            """,
            (_db_time(now), event_id),
        )
        if not ok:
            raise StorageOperationError()
        return bool(affected)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _db_time(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _aware(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _summary_from_row(row: dict[str, Any]) -> SessionSummary:
    return SessionSummary(
        summary_id=str(row["summary_id"]), user_id=str(row["user_id"]), session_id=str(row["session_id"]),
        version=int(row["version"]), summary_text=str(row["summary_text"]),
        structured_data=json.loads(row.get("structured_json") or "{}"),
        covers_until_message_id=int(row["covers_until_message_id"]),
        created_at=_aware(row["created_at"]), updated_at=_aware(row["updated_at"]),
    )


def _item_from_row(row: dict[str, Any]) -> MemoryItem:
    return MemoryItem(
        memory_id=str(row["memory_id"]), user_id=str(row["user_id"]),
        memory_type=str(row["memory_type"]), memory_key=str(row["memory_key"]), content=str(row["content"]),
        structured_data=json.loads(row.get("structured_json") or "{}"), confidence=float(row["confidence"]),
        status=str(row["status"]), version=int(row["version"]), supersedes_id=row.get("supersedes_id"),
        valid_from=_aware(row.get("valid_from")), valid_until=_aware(row.get("valid_until")),
        expires_at=_aware(row.get("expires_at")), created_at=_aware(row["created_at"]),
        updated_at=_aware(row["updated_at"]),
    )


def _item_args(item: MemoryItem) -> tuple[Any, ...]:
    return (
        str(item.memory_id), item.user_id, item.memory_type.value, item.memory_key, item.content,
        _json(item.structured_data), item.confidence, item.status.value, item.version,
        str(item.supersedes_id) if item.supersedes_id else None, _db_time(item.valid_from),
        _db_time(item.valid_until), _db_time(item.expires_at), _db_time(item.created_at), _db_time(item.updated_at),
    )


def _outbox_from_row(row: dict[str, Any]) -> MemoryOutboxEvent:
    return MemoryOutboxEvent(
        event_id=str(row["event_id"]), event_type=str(row["event_type"]), aggregate_id=str(row["aggregate_id"]),
        payload=json.loads(row.get("payload_json") or "{}"), status=str(row["status"]),
        attempts=int(row.get("attempts", 0)), available_at=_aware(row["available_at"]),
        lease_until=_aware(row.get("lease_until")), last_error_code=row.get("last_error_code"),
        created_at=_aware(row["created_at"]), processed_at=_aware(row.get("processed_at")),
    )
