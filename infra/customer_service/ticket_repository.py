from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from domain.customer_service_agent.interfaces.i_ticket_repository import ITicketRepository
from infra.db.mysql_client import MySQLClient, get_mysql_client
from pkg.exceptions.exception import StorageOperationError, StorageUnavailableError, ToolValidationError
from pkg.security import get_local_user_id

TICKET_TYPES = {"refund", "claim", "account_open", "account_change", "complaint", "general"}
TICKET_PRIORITIES = {"low", "medium", "high", "urgent"}
TICKET_STATUSES = {"created", "processing", "pending_review", "resolved", "closed", "escalated"}
ALLOWED_STATUS_TRANSITIONS = {
    "created": {"processing", "pending_review", "escalated", "closed"},
    "processing": {"pending_review", "resolved", "escalated", "closed"},
    "pending_review": {"processing", "resolved", "escalated", "closed"},
    "resolved": {"closed"},
    "escalated": {"processing", "resolved", "closed"},
    "closed": set(),
}


class TicketRepository(ITicketRepository):
    """MySQL-only durable ticket repository for local single-user mode."""

    def __init__(self, mysql_client: Optional[MySQLClient] = None, user_id: str | None = None):
        self.mysql_client = mysql_client
        self.user_id = user_id or get_local_user_id()
        self._ready = mysql_client is not None and self._ensure_table()

    @property
    def available(self) -> bool:
        if self.mysql_client is None:
            self.mysql_client = get_mysql_client()
        if self.mysql_client is not None and not self._ready:
            self._ready = self._ensure_table()
        return self.mysql_client is not None and self._ready

    def _ensure_table(self) -> bool:
        assert self.mysql_client is not None
        ok, _ = self.mysql_client.execute_update(
            """
            CREATE TABLE IF NOT EXISTS cs_tickets (
                ticket_id VARCHAR(64) PRIMARY KEY,
                operation_key VARCHAR(128) NULL,
                user_id VARCHAR(128) NOT NULL,
                type VARCHAR(64) NOT NULL,
                priority VARCHAR(32) NOT NULL,
                status VARCHAR(32) NOT NULL,
                summary VARCHAR(255) NOT NULL,
                details TEXT NOT NULL,
                created_at VARCHAR(64) NOT NULL,
                updated_at VARCHAR(64) NOT NULL,
                UNIQUE KEY uk_cs_tickets_operation_key (operation_key),
                INDEX idx_cs_tickets_user_id (user_id, created_at)
            ) CHARACTER SET utf8mb4
            """
        )
        if not ok:
            return False
        ok, row = self.mysql_client.execute_query(
            """
            SELECT COUNT(*) AS count
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'cs_tickets' AND COLUMN_NAME = 'operation_key'
            """,
            fetch_one=True,
        )
        if not ok:
            return False
        if not int(row["count"]):
            ok, _ = self.mysql_client.execute_update(
                "ALTER TABLE cs_tickets ADD COLUMN operation_key VARCHAR(128) NULL AFTER ticket_id"
            )
            if not ok:
                return False
        ok, row = self.mysql_client.execute_query(
            """
            SELECT COUNT(*) AS count
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'cs_tickets'
              AND INDEX_NAME = 'uk_cs_tickets_operation_key'
            """,
            fetch_one=True,
        )
        if not ok:
            return False
        if not int(row["count"]):
            ok, _ = self.mysql_client.execute_update(
                "CREATE UNIQUE INDEX uk_cs_tickets_operation_key ON cs_tickets (operation_key)"
            )
        return ok

    def _require_available(self) -> MySQLClient:
        if not self.available or self.mysql_client is None:
            raise StorageUnavailableError()
        return self.mysql_client

    def create(
        self,
        ticket_type: str,
        priority: str,
        summary: str,
        details: str,
        user_id: str,
        operation_key: str | None = None,
    ) -> dict[str, Any]:
        client = self._require_available()
        _validate_ticket_create(ticket_type, priority, summary, details, operation_key)
        if operation_key:
            existing = self._query_by_operation_key(operation_key)
            if existing is not None:
                return existing

        now = datetime.now(timezone.utc).isoformat()
        ticket = {
            "ticket_id": f"TK-{uuid4().hex.upper()}",
            "operation_key": operation_key,
            "user_id": self.user_id,
            "type": ticket_type,
            "priority": priority,
            "status": "created",
            "summary": summary,
            "details": details,
            "created_at": now,
            "updated_at": now,
        }
        ok, _ = client.execute_update(
            """
            INSERT INTO cs_tickets
                (ticket_id, operation_key, user_id, type, priority, status, summary, details, created_at, updated_at)
            VALUES
                (%(ticket_id)s, %(operation_key)s, %(user_id)s, %(type)s, %(priority)s,
                 %(status)s, %(summary)s, %(details)s, %(created_at)s, %(updated_at)s)
            """,
            ticket,
        )
        if not ok:
            if operation_key:
                existing = self._query_by_operation_key(operation_key)
                if existing is not None:
                    return existing
            raise StorageOperationError()
        return ticket

    def query(self, ticket_id: str) -> Optional[dict[str, Any]]:
        client = self._require_available()
        if not ticket_id or len(ticket_id) > 64:
            raise ToolValidationError()
        ok, result = client.execute_query(
            "SELECT * FROM cs_tickets WHERE ticket_id = %s AND user_id = %s",
            (ticket_id, self.user_id),
            fetch_one=True,
        )
        if not ok:
            raise StorageOperationError()
        return result or None

    def update_status(self, ticket_id: str, status: str) -> Optional[dict[str, Any]]:
        if status not in TICKET_STATUSES:
            raise ToolValidationError()
        ticket = self.query(ticket_id)
        if ticket is None:
            return None
        current_status = str(ticket["status"])
        if current_status == status:
            return ticket
        if status not in ALLOWED_STATUS_TRANSITIONS.get(current_status, set()):
            raise ToolValidationError()

        updated_at = datetime.now(timezone.utc).isoformat()
        client = self._require_available()
        ok, affected = client.execute_update(
            """
            UPDATE cs_tickets
            SET status = %s, updated_at = %s
            WHERE ticket_id = %s AND user_id = %s AND status = %s
            """,
            (status, updated_at, ticket_id, self.user_id, current_status),
        )
        if not ok:
            raise StorageOperationError()
        if int(affected) != 1:
            # A concurrent writer won. Return its result only if the desired
            # idempotent target was reached; otherwise report a failed write.
            current = self.query(ticket_id)
            if current is not None and current.get("status") == status:
                return current
            raise StorageOperationError()
        return {**ticket, "status": status, "updated_at": updated_at}

    def _query_by_operation_key(self, operation_key: str) -> dict[str, Any] | None:
        client = self._require_available()
        ok, result = client.execute_query(
            "SELECT * FROM cs_tickets WHERE operation_key = %s AND user_id = %s",
            (operation_key, self.user_id),
            fetch_one=True,
        )
        if not ok:
            raise StorageOperationError()
        return result or None


class InMemoryTicketRepository(ITicketRepository):
    """Explicit test double; production bootstrap never uses this repository."""

    def __init__(self, user_id: str | None = None) -> None:
        self.user_id = user_id or get_local_user_id()
        self._tickets: dict[str, dict[str, Any]] = {}
        self._operations: dict[str, str] = {}
        self._lock = threading.RLock()

    def create(self, ticket_type, priority, summary, details, user_id, operation_key=None):
        _validate_ticket_create(ticket_type, priority, summary, details, operation_key)
        with self._lock:
            if operation_key and operation_key in self._operations:
                return dict(self._tickets[self._operations[operation_key]])
            now = datetime.now(timezone.utc).isoformat()
            ticket = {
                "ticket_id": f"TK-{uuid4().hex.upper()}",
                "operation_key": operation_key,
                "user_id": self.user_id,
                "type": ticket_type,
                "priority": priority,
                "status": "created",
                "summary": summary,
                "details": details,
                "created_at": now,
                "updated_at": now,
            }
            self._tickets[ticket["ticket_id"]] = ticket
            if operation_key:
                self._operations[operation_key] = ticket["ticket_id"]
            return dict(ticket)

    def query(self, ticket_id):
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            return dict(ticket) if ticket else None

    def update_status(self, ticket_id, status):
        if status not in TICKET_STATUSES:
            raise ToolValidationError()
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if ticket is None:
                return None
            current = str(ticket["status"])
            if current != status and status not in ALLOWED_STATUS_TRANSITIONS.get(current, set()):
                raise ToolValidationError()
            ticket["status"] = status
            ticket["updated_at"] = datetime.now(timezone.utc).isoformat()
            return dict(ticket)


def _validate_ticket_create(
    ticket_type: str,
    priority: str,
    summary: str,
    details: str,
    operation_key: str | None,
) -> None:
    if ticket_type not in TICKET_TYPES or priority not in TICKET_PRIORITIES:
        raise ToolValidationError()
    if not summary.strip() or len(summary) > 255:
        raise ToolValidationError()
    if not details.strip() or len(details) > 65_535:
        raise ToolValidationError()
    if operation_key is not None and (not operation_key or len(operation_key) > 128):
        raise ToolValidationError()
