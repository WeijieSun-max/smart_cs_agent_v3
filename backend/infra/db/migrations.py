from __future__ import annotations

import hashlib
from pathlib import Path

from infra.db.mysql_client import MySQLClient
from pkg.log.logger import get_logger

logger = get_logger()


def apply_migrations(client: MySQLClient | None, root: Path) -> None:
    if client is None:
        return
    client.execute_update(
        """CREATE TABLE IF NOT EXISTS cs_schema_migrations (
        version VARCHAR(128) PRIMARY KEY, checksum CHAR(64) NOT NULL, applied_at DATETIME(6) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin"""
    )
    for path in sorted(root.glob("*.sql")):
        content = path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        ok, row = client.execute_query("SELECT checksum FROM cs_schema_migrations WHERE version=%s", (path.name,), fetch_one=True)
        if ok and row:
            if row["checksum"] != checksum:
                raise RuntimeError(f"migration checksum mismatch: {path.name}")
            continue
        for statement in (item.strip() for item in content.split(";")):
            if statement:
                success, _ = client.execute_update(statement)
                if not success:
                    raise RuntimeError(f"migration failed: {path.name}")
        success, _ = client.execute_update("INSERT INTO cs_schema_migrations(version,checksum,applied_at) VALUES (%s,%s,UTC_TIMESTAMP(6))", (path.name, checksum))
        if not success:
            raise RuntimeError(f"migration record failed: {path.name}")
        logger.info("Applied migration {}", path.name)
    _reconcile_session_schema(client)


def _reconcile_session_schema(client: MySQLClient) -> None:
    """Keep legacy workbench and governed-identity session columns compatible."""
    columns = {
        "title": "ALTER TABLE cs_sessions ADD COLUMN title VARCHAR(255) NOT NULL DEFAULT '新会话' AFTER user_id",
        "agent_id": "ALTER TABLE cs_sessions ADD COLUMN agent_id VARCHAR(64) NOT NULL DEFAULT 'general' AFTER title",
        "favorite": "ALTER TABLE cs_sessions ADD COLUMN favorite TINYINT(1) NOT NULL DEFAULT 0 AFTER agent_id",
        "message_count": "ALTER TABLE cs_sessions ADD COLUMN message_count INT NOT NULL DEFAULT 0 AFTER favorite",
        "status": "ALTER TABLE cs_sessions ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT 'active' AFTER message_count",
        "version": "ALTER TABLE cs_sessions ADD COLUMN version BIGINT UNSIGNED NOT NULL DEFAULT 1 AFTER status",
    }
    for column, ddl in columns.items():
        ok, row = client.execute_query(
            """SELECT COUNT(*) AS count FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='cs_sessions' AND COLUMN_NAME=%s""",
            (column,), fetch_one=True,
        )
        if not ok or row is None:
            raise RuntimeError("failed to inspect cs_sessions schema")
        if not int(row["count"]):
            success, _ = client.execute_update(ddl)
            if not success:
                raise RuntimeError(f"failed to add cs_sessions.{column}")
