from __future__ import annotations

from datetime import datetime, timezone
import json
import uuid
from typing import TYPE_CHECKING, Any

from domain.customer_service_agent.interfaces.i_conversation_archive import IConversationArchive
from infra.db.mysql_client import MySQLClient, get_mysql_client
from pkg.exceptions.exception import StorageOperationError, StorageUnavailableError
from pkg.security import get_local_user_id
from pkg.telemetry import record_fallback

if TYPE_CHECKING:
    from infra.memory.mysql_memory_repository import MySQLMemoryRepository


class MySQLConversationArchive(IConversationArchive):
    def __init__(
        self,
        mysql_client: MySQLClient | None,
        user_id: str | None = None,
        memory_repository: "MySQLMemoryRepository | None" = None,
    ) -> None:
        self.mysql_client = mysql_client
        self._default_user_id = user_id or get_local_user_id()
        self.memory_repository = memory_repository
        self._ready = False
        if mysql_client is not None:
            self._ready = self._ensure_tables()

    def _scope(self, user_id: str | None) -> str:
        return user_id or self._default_user_id

    @property
    def available(self) -> bool:
        if self.mysql_client is None:
            self.mysql_client = get_mysql_client()
            if self.mysql_client is None:
                return False
        if not self._ready:
            self._ready = self._ensure_tables()
        return self._ready

    def _ensure_tables(self) -> bool:
        assert self.mysql_client is not None
        statements = [
            """
            CREATE TABLE IF NOT EXISTS cs_sessions (
                session_id VARCHAR(128) PRIMARY KEY,
                user_id VARCHAR(128) NOT NULL DEFAULT 'local-user',
                title VARCHAR(255) NOT NULL,
                agent_id VARCHAR(64) NOT NULL,
                favorite TINYINT(1) NOT NULL DEFAULT 0,
                message_count INT NOT NULL DEFAULT 0,
                created_at DATETIME(6) NOT NULL,
                updated_at DATETIME(6) NOT NULL,
                INDEX idx_cs_sessions_updated_at (updated_at)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS cs_messages (
                message_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(128) NOT NULL,
                turn_id VARCHAR(64) NULL,
                role VARCHAR(32) NOT NULL,
                content LONGTEXT NOT NULL,
                created_at DATETIME(6) NOT NULL,
                INDEX idx_cs_messages_session_id (session_id, message_id),
                UNIQUE KEY uk_cs_messages_turn_role (turn_id, role),
                CONSTRAINT fk_cs_messages_session
                    FOREIGN KEY (session_id) REFERENCES cs_sessions(session_id)
                    ON DELETE CASCADE
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS cs_agent_runs (
                run_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                turn_id VARCHAR(64) NOT NULL UNIQUE,
                session_id VARCHAR(128) NOT NULL,
                status VARCHAR(32) NOT NULL,
                stop_requested TINYINT(1) NOT NULL DEFAULT 0,
                started_at DATETIME(6) NOT NULL,
                completed_at DATETIME(6) NULL,
                INDEX idx_cs_agent_runs_session_id (session_id, started_at),
                CONSTRAINT fk_cs_agent_runs_session
                    FOREIGN KEY (session_id) REFERENCES cs_sessions(session_id)
                    ON DELETE CASCADE
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS cs_agent_run_steps (
                step_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                turn_id VARCHAR(64) NOT NULL,
                sequence_no INT NOT NULL,
                step_type VARCHAR(32) NOT NULL,
                status VARCHAR(32) NOT NULL,
                title VARCHAR(500) NOT NULL,
                payload_json LONGTEXT NULL,
                created_at DATETIME(6) NOT NULL,
                UNIQUE KEY uk_cs_agent_run_steps_turn_sequence (turn_id, sequence_no),
                INDEX idx_cs_agent_run_steps_turn_id (turn_id, step_id),
                CONSTRAINT fk_cs_agent_run_steps_turn
                    FOREIGN KEY (turn_id) REFERENCES cs_agent_runs(turn_id)
                    ON DELETE CASCADE
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
        ]
        if not all(self.mysql_client.execute_update(sql)[0] for sql in statements):
            return False
        return self._ensure_columns_and_indexes()

    def _ensure_columns_and_indexes(self) -> bool:
        assert self.mysql_client is not None
        migrations = [
            (
                "cs_sessions",
                "user_id",
                "ALTER TABLE cs_sessions ADD COLUMN user_id VARCHAR(128) NOT NULL DEFAULT 'local-user' AFTER session_id",
            ),
            (
                "cs_messages",
                "turn_id",
                "ALTER TABLE cs_messages ADD COLUMN turn_id VARCHAR(64) NULL AFTER session_id",
            ),
        ]
        for table, column, sql in migrations:
            ok, row = self.mysql_client.execute_query(
                """
                SELECT COUNT(*) AS count
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s
                """,
                (table, column),
                fetch_one=True,
            )
            if not ok or (not int(row["count"]) and not self.mysql_client.execute_update(sql)[0]):
                return False

        index_migrations = [
            (
                "cs_sessions",
                "idx_cs_sessions_user_updated",
                "CREATE INDEX idx_cs_sessions_user_updated ON cs_sessions (user_id, updated_at)",
            ),
            (
                "cs_messages",
                "uk_cs_messages_turn_role",
                "CREATE UNIQUE INDEX uk_cs_messages_turn_role ON cs_messages (turn_id, role)",
            ),
        ]
        for table, index_name, sql in index_migrations:
            ok, row = self.mysql_client.execute_query(
                """
                SELECT COUNT(*) AS count
                FROM information_schema.STATISTICS
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND INDEX_NAME = %s
                """,
                (table, index_name),
                fetch_one=True,
            )
            if not ok or (not int(row["count"]) and not self.mysql_client.execute_update(sql)[0]):
                return False
        ok, _ = self.mysql_client.execute_update(
            "UPDATE cs_sessions SET user_id = %s WHERE user_id = 'local-user'",
            (self._default_user_id,),
        )
        if not ok:
            return False
        return True

    def _require_available(self) -> MySQLClient:
        if not self.available or self.mysql_client is None:
            raise StorageUnavailableError()
        return self.mysql_client

    def create_session(
        self,
        session_id: str,
        title: str,
        agent_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, object] | None:
        scope = self._scope(user_id)
        client = self._require_available()
        now = _utc_now()
        ok, _ = client.execute_update(
            """
            INSERT IGNORE INTO cs_sessions
                (session_id, user_id, title, agent_id, favorite, message_count, created_at, updated_at)
            VALUES (%s, %s, %s, %s, 0, 0, %s, %s)
            """,
            (session_id, scope, title.strip() or "新会话", agent_id, now, now),
        )
        if not ok:
            raise StorageOperationError()
        return self.get_session(session_id, user_id=scope)

    def get_session(self, session_id: str, *, user_id: str | None = None) -> dict[str, object] | None:
        scope = self._scope(user_id)
        client = self._require_available()
        ok, row = client.execute_query(
            """
            SELECT session_id, title, agent_id, favorite, message_count, created_at, updated_at
            FROM cs_sessions WHERE session_id = %s AND user_id = %s
            """,
            (session_id, scope),
            fetch_one=True,
        )
        if not ok:
            raise StorageOperationError()
        return _session_from_row(row) if row else None

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: str,
        turn_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> bool:
        scope = self._scope(user_id)
        client = self._require_available()
        created_at = _parse_timestamp(timestamp)
        title = _title_from_message(content)
        def operation(cursor) -> int:
            cursor.execute(
                """
                INSERT IGNORE INTO cs_sessions
                    (session_id, user_id, title, agent_id, favorite, message_count, created_at, updated_at)
                VALUES (%s, %s, %s, 'general', 0, 0, %s, %s)
                """,
                (session_id, scope, "新会话", created_at, created_at),
            )
            inserted = cursor.execute(
                """
                INSERT IGNORE INTO cs_messages (session_id, turn_id, role, content, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, turn_id, role, content, created_at),
            )
            if not inserted:
                return 0
            cursor.execute(
                """
                UPDATE cs_sessions
                SET title = CASE WHEN message_count = 0 AND %s = 'user' THEN %s ELSE title END,
                    message_count = message_count + 1,
                    updated_at = %s
                WHERE session_id = %s AND user_id = %s
                """,
                (role, title, created_at, session_id, scope),
            )
            return inserted

        ok, _ = client.execute_in_transaction(operation)
        if not ok:
            raise StorageOperationError()
        return True

    def complete_turn(
        self,
        session_id: str,
        content: str,
        timestamp: str,
        turn_id: str,
        *,
        enqueue_memory: bool = False,
        user_id: str | None = None,
        fencing_token: int | None = None,
        lease_owner_id: str | None = None,
    ) -> bool:
        scope = self._scope(user_id)
        client = self._require_available()
        created_at = _parse_timestamp(timestamp)

        def operation(cursor) -> bool:
            if fencing_token is not None:
                cursor.execute(
                    """
                    SELECT fencing_token FROM cs_turn_leases
                    WHERE user_id = %s AND session_id = %s AND turn_id = %s
                      AND owner_id = %s AND fencing_token = %s
                      AND lease_until > UTC_TIMESTAMP(6) AND stop_requested = 0
                    FOR UPDATE
                    """,
                    (scope, session_id, turn_id, lease_owner_id, fencing_token),
                )
                if cursor.fetchone() is None:
                    raise RuntimeError("stale turn lease")
            cursor.execute(
                """
                INSERT IGNORE INTO cs_sessions
                    (session_id, user_id, title, agent_id, favorite, message_count, created_at, updated_at)
                VALUES (%s, %s, '新会话', 'general', 0, 0, %s, %s)
                """,
                (session_id, scope, created_at, created_at),
            )
            inserted = cursor.execute(
                """
                INSERT IGNORE INTO cs_messages (session_id, turn_id, role, content, created_at)
                VALUES (%s, %s, 'assistant', %s, %s)
                """,
                (session_id, turn_id, content, created_at),
            )
            if inserted:
                cursor.execute(
                    """
                    UPDATE cs_sessions
                    SET message_count = message_count + 1, updated_at = %s
                    WHERE session_id = %s AND user_id = %s
                    """,
                    (created_at, session_id, scope),
                )
            if enqueue_memory:
                payload = json.dumps(
                    {"user_id": scope, "session_id": session_id, "turn_id": turn_id},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                cursor.execute(
                    """
                    INSERT IGNORE INTO cs_memory_outbox
                        (event_id, event_type, aggregate_id, payload_json, status, attempts, available_at, created_at)
                    VALUES (%s, 'turn_completed', %s, %s, 'pending', 0, %s, %s)
                    """,
                    (str(uuid.uuid4()), turn_id, payload, created_at, created_at),
                )
            return True

        ok, result = client.execute_in_transaction(operation)
        if not ok:
            raise StorageOperationError()
        return bool(result)

    def get_message_by_turn(
        self,
        turn_id: str,
        role: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, str] | None:
        scope = self._scope(user_id)
        client = self._require_available()
        ok, row = client.execute_query(
            """
            SELECT m.session_id, m.role, m.content, m.created_at, m.turn_id
            FROM cs_messages m
            JOIN cs_sessions s ON s.session_id = m.session_id
            WHERE m.turn_id = %s AND m.role = %s AND s.user_id = %s
            """,
            (turn_id, role, scope),
            fetch_one=True,
        )
        if not ok:
            raise StorageOperationError()
        if not row:
            return None
        return {
            "session_id": str(row["session_id"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "timestamp": _isoformat(row["created_at"]),
            "turn_id": str(row["turn_id"]),
        }

    def get_history(
        self,
        session_id: str,
        last_n: int,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, str]]:
        scope = self._scope(user_id)
        client = self._require_available()
        ok, rows = client.execute_query(
            """
            SELECT m.session_id, m.role, m.content, m.created_at, m.turn_id
            FROM cs_messages m
            JOIN cs_sessions s ON s.session_id = m.session_id
            WHERE m.session_id = %s AND s.user_id = %s
            ORDER BY message_id DESC
            LIMIT %s
            """,
            (session_id, scope, last_n),
        )
        if not ok:
            record_fallback("mysql_conversation_archive")
            raise StorageOperationError()
        return [
            {
                "session_id": str(row["session_id"]),
                "role": str(row["role"]),
                "content": str(row["content"]),
                "timestamp": _isoformat(row["created_at"]),
                **({"turn_id": str(row["turn_id"])} if row.get("turn_id") else {}),
            }
            for row in reversed(rows)
        ]

    def list_sessions(self, *, user_id: str | None = None) -> list[dict[str, object]]:
        scope = self._scope(user_id)
        client = self._require_available()
        ok, rows = client.execute_query(
            """
            SELECT session_id, title, agent_id, favorite, message_count, created_at, updated_at
            FROM cs_sessions
            WHERE user_id = %s
            ORDER BY updated_at DESC
            """,
            (scope,),
        )
        if not ok:
            record_fallback("mysql_conversation_archive")
            raise StorageOperationError()
        return [_session_from_row(row) for row in rows]

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        favorite: bool | None = None,
        user_id: str | None = None,
    ) -> dict[str, object] | None:
        scope = self._scope(user_id)
        client = self._require_available()
        assignments: list[str] = []
        values: list[object] = []
        if title is not None:
            assignments.append("title = %s")
            values.append(title.strip() or "新会话")
        if favorite is not None:
            assignments.append("favorite = %s")
            values.append(1 if favorite else 0)
        if assignments:
            assignments.append("updated_at = %s")
            values.append(_utc_now())
            values.extend([session_id, scope])
            ok, _ = client.execute_update(
                f"UPDATE cs_sessions SET {', '.join(assignments)} WHERE session_id = %s AND user_id = %s",
                tuple(values),
            )
            if not ok:
                raise StorageOperationError()
        return self.get_session(session_id, user_id=scope)

    def delete_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        scope = self._scope(user_id)
        client = self._require_available()
        memory_repository = getattr(self, "memory_repository", None)
        if memory_repository is None:
            ok, affected = client.execute_update(
                "DELETE FROM cs_sessions WHERE session_id = %s AND user_id = %s",
                (session_id, scope),
            )
            if not ok:
                raise StorageOperationError()
            return bool(affected)

        def operation(cursor) -> bool:
            cursor.execute(
                "SELECT session_id FROM cs_sessions WHERE session_id = %s AND user_id = %s FOR UPDATE",
                (session_id, scope),
            )
            if cursor.fetchone() is None:
                return False
            memory_repository.delete_session_sources_in_transaction(
                cursor,
                scope,
                session_id,
            )
            cursor.execute(
                "DELETE FROM cs_sessions WHERE session_id = %s AND user_id = %s",
                (session_id, scope),
            )
            return True

        ok, affected = client.execute_in_transaction(operation)
        if not ok:
            raise StorageOperationError()
        return bool(affected)

    def start_run(
        self,
        session_id: str,
        turn_id: str,
        started_at: str,
        *,
        user_id: str | None = None,
    ) -> None:
        scope = self._scope(user_id)
        client = self._require_available()
        timestamp = _parse_timestamp(started_at)
        ok, _ = client.execute_transaction([
            (
                """
                INSERT IGNORE INTO cs_sessions
                    (session_id, user_id, title, agent_id, favorite, message_count, created_at, updated_at)
                VALUES (%s, %s, '新会话', 'general', 0, 0, %s, %s)
                """,
                (session_id, scope, timestamp, timestamp),
            ),
            (
                """
                INSERT INTO cs_agent_runs
                    (turn_id, session_id, status, stop_requested, started_at)
                VALUES (%s, %s, 'running', 0, %s)
                ON DUPLICATE KEY UPDATE status = 'running', stop_requested = 0,
                    started_at = VALUES(started_at), completed_at = NULL
                """,
                (turn_id, session_id, timestamp),
            ),
        ])
        if not ok:
            raise StorageOperationError()

    def mark_run_stop_requested(self, turn_id: str) -> None:
        client = self._require_available()
        ok, _ = client.execute_update(
            "UPDATE cs_agent_runs SET stop_requested = 1 WHERE turn_id = %s",
            (turn_id,),
        )
        if not ok:
            raise StorageOperationError()

    def finish_run(self, turn_id: str, status: str, completed_at: str) -> None:
        client = self._require_available()
        ok, _ = client.execute_update(
            "UPDATE cs_agent_runs SET status = %s, completed_at = %s WHERE turn_id = %s",
            (status, _parse_timestamp(completed_at), turn_id),
        )
        if not ok:
            raise StorageOperationError()

    def record_run_step(
        self,
        turn_id: str,
        sequence_no: int,
        step_type: str,
        status: str,
        title: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        client = self._require_available()
        ok, _ = client.execute_update(
            """
            INSERT INTO cs_agent_run_steps
                (turn_id, sequence_no, step_type, status, title, payload_json, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE step_type = VALUES(step_type), status = VALUES(status),
                title = VALUES(title), payload_json = VALUES(payload_json)
            """,
            (
                turn_id,
                sequence_no,
                step_type,
                status,
                title[:500],
                json.dumps(payload, ensure_ascii=False),
                _parse_timestamp(created_at),
            ),
        )
        if not ok:
            raise StorageOperationError()

    def list_runs(
        self,
        session_id: str,
        limit: int = 50,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        scope = self._scope(user_id)
        client = self._require_available()
        ok, runs = client.execute_query(
            """
            SELECT r.turn_id, r.session_id, r.status, r.stop_requested, r.started_at, r.completed_at
            FROM cs_agent_runs r
            JOIN cs_sessions s ON s.session_id = r.session_id
            WHERE r.session_id = %s AND s.user_id = %s
            ORDER BY started_at DESC
            LIMIT %s
            """,
            (session_id, scope, limit),
        )
        if not ok:
            raise StorageOperationError()
        turn_ids = [str(run["turn_id"]) for run in runs]
        steps_by_turn: dict[str, list[dict[str, Any]]] = {turn_id: [] for turn_id in turn_ids}
        if turn_ids:
            placeholders = ", ".join(["%s"] * len(turn_ids))
            steps_ok, steps = client.execute_query(
                f"""
                SELECT turn_id, sequence_no, step_type, status, title, payload_json, created_at
                FROM cs_agent_run_steps
                WHERE turn_id IN ({placeholders})
                ORDER BY turn_id, sequence_no ASC
                """,
                tuple(turn_ids),
            )
            if not steps_ok:
                raise StorageOperationError()
            for step in steps:
                try:
                    payload = json.loads(step.get("payload_json") or "{}")
                except (json.JSONDecodeError, TypeError):
                    payload = {}
                turn_id = str(step["turn_id"])
                steps_by_turn.setdefault(turn_id, []).append({
                    "id": f"{turn_id}-{step['sequence_no']}",
                    "type": step["step_type"],
                    "status": step["status"],
                    "title": step["title"],
                    "toolName": payload.get("node_name"),
                    "description": payload.get("description"),
                    "input": payload.get("input"),
                    "output": payload.get("output"),
                    "startedAt": _isoformat(step["created_at"]),
                    "completedAt": payload.get("completed_at"),
                    "duration": payload.get("duration_ms"),
                    "tokenUsage": payload.get("token_usage"),
                    "modelCalls": payload.get("model_calls", 0),
                })

        result: list[dict[str, Any]] = []
        for run in runs:
            mapped_steps = steps_by_turn.get(str(run["turn_id"]), [])
            result.append({
                "turn_id": str(run["turn_id"]),
                "session_id": str(run["session_id"]),
                "status": str(run["status"]),
                "stop_requested": bool(run["stop_requested"]),
                "started_at": _isoformat(run["started_at"]),
                "completed_at": _isoformat(run["completed_at"]) if run["completed_at"] else None,
                "steps": mapped_steps,
            })
        return result

def _session_from_row(row: dict[str, Any]) -> dict[str, object]:
    return {
        "id": str(row["session_id"]),
        "title": str(row["title"]),
        "agent_id": str(row["agent_id"]),
        "created_at": _isoformat(row["created_at"]),
        "updated_at": _isoformat(row["updated_at"]),
        "favorite": bool(row["favorite"]),
        "message_count": int(row["message_count"]),
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _isoformat(value: object) -> str:
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc).isoformat()
    return str(value)


def _title_from_message(content: str) -> str:
    title = " ".join(content.strip().split())
    return title[:40] + ("…" if len(title) > 40 else "") or "新会话"
