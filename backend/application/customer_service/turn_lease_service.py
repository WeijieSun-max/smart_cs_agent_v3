from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from pkg.config.settings import get_settings
from pkg.exceptions.exception import StorageOperationError, StorageUnavailableError

if TYPE_CHECKING:
    from infra.db.mysql_client import MySQLClient


@dataclass
class TurnLease:
    user_id: str
    session_id: str
    turn_id: str
    owner_id: str
    fencing_token: int
    stop_requested: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def checkpoint_namespace(self) -> str:
        return f"turn:{self.fencing_token}" if self.fencing_token > 0 else ""


class TurnLeaseManager:
    """MySQL-backed session lease with fencing and durable cancellation."""

    def __init__(self, client: "MySQLClient | None", *, owner_id: str | None = None) -> None:
        self.client = client
        self.owner_id = owner_id or f"{os.getpid()}-{uuid.uuid4().hex}"[:64]

    @property
    def available(self) -> bool:
        return self.client is not None

    @asynccontextmanager
    async def hold(self, user_id: str, session_id: str, turn_id: str):
        settings = get_settings()
        if not settings.turn_distributed_lease_enabled:
            yield TurnLease(user_id, session_id, turn_id, self.owner_id, 0)
            return
        lease = await asyncio.to_thread(self.acquire, user_id, session_id, turn_id)
        owner_task = asyncio.current_task()
        heartbeat = asyncio.create_task(
            self._heartbeat(lease, owner_task),
            name=f"turn-lease-{turn_id}",
        )
        try:
            yield lease
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            await asyncio.shield(asyncio.to_thread(self.release, lease))

    def acquire(self, user_id: str, session_id: str, turn_id: str) -> TurnLease:
        client = self._require_client()
        now = _utc_now()
        lease_until = now + timedelta(seconds=get_settings().turn_lease_seconds)

        def operation(cursor):
            cursor.execute(
                """
                SELECT turn_id, owner_id, fencing_token, lease_until
                FROM cs_turn_leases
                WHERE user_id = %s AND session_id = %s
                FOR UPDATE
                """,
                (user_id, session_id),
            )
            current = cursor.fetchone()
            if current is not None and _as_utc_naive(current["lease_until"]) > now:
                return None
            fencing_token = int(current["fencing_token"]) + 1 if current is not None else 1
            if current is None:
                cursor.execute(
                    """
                    INSERT INTO cs_turn_leases
                        (user_id, session_id, turn_id, owner_id, fencing_token,
                         lease_until, stop_requested, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, 0, %s)
                    """,
                    (user_id, session_id, turn_id, self.owner_id, fencing_token, lease_until, now),
                )
            else:
                cursor.execute(
                    """
                    UPDATE cs_turn_leases
                    SET turn_id = %s, owner_id = %s, fencing_token = %s,
                        lease_until = %s, stop_requested = 0, updated_at = %s
                    WHERE user_id = %s AND session_id = %s
                    """,
                    (turn_id, self.owner_id, fencing_token, lease_until, now, user_id, session_id),
                )
            return TurnLease(user_id, session_id, turn_id, self.owner_id, fencing_token)

        ok, result = client.execute_in_transaction(operation)
        if not ok:
            raise StorageOperationError()
        if result is None:
            raise RuntimeError(f"Session {session_id} already has a running agent")
        return result

    def renew(self, lease: TurnLease) -> str:
        client = self._require_client()
        now = _utc_now()
        lease_until = now + timedelta(seconds=get_settings().turn_lease_seconds)

        def operation(cursor):
            affected = cursor.execute(
                """
                UPDATE cs_turn_leases
                SET lease_until = %s, updated_at = %s
                WHERE user_id = %s AND session_id = %s AND turn_id = %s
                  AND owner_id = %s AND fencing_token = %s
                """,
                (
                    lease_until,
                    now,
                    lease.user_id,
                    lease.session_id,
                    lease.turn_id,
                    lease.owner_id,
                    lease.fencing_token,
                ),
            )
            if affected != 1:
                return "lost"
            cursor.execute(
                """
                SELECT stop_requested FROM cs_turn_leases
                WHERE user_id = %s AND session_id = %s AND turn_id = %s
                  AND owner_id = %s AND fencing_token = %s
                """,
                (
                    lease.user_id,
                    lease.session_id,
                    lease.turn_id,
                    lease.owner_id,
                    lease.fencing_token,
                ),
            )
            row = cursor.fetchone()
            return "stop" if row is not None and bool(row["stop_requested"]) else "active"

        ok, result = client.execute_in_transaction(operation)
        if not ok:
            return "lost"
        return str(result)

    def release(self, lease: TurnLease) -> None:
        if self.client is None or lease.fencing_token <= 0:
            return
        now = _utc_now()
        ok, _ = self.client.execute_update(
            """
            UPDATE cs_turn_leases
            SET lease_until = %s, updated_at = %s
            WHERE user_id = %s AND session_id = %s AND turn_id = %s
              AND owner_id = %s AND fencing_token = %s
            """,
            (
                now,
                now,
                lease.user_id,
                lease.session_id,
                lease.turn_id,
                lease.owner_id,
                lease.fencing_token,
            ),
        )
        if not ok:
            raise StorageOperationError()

    def request_stop(self, user_id: str, session_id: str) -> bool:
        if self.client is None:
            return False
        ok, affected = self.client.execute_update(
            """
            UPDATE cs_turn_leases
            SET stop_requested = 1, updated_at = UTC_TIMESTAMP(6)
            WHERE user_id = %s AND session_id = %s
              AND lease_until > UTC_TIMESTAMP(6)
            """,
            (user_id, session_id),
        )
        if not ok:
            raise StorageOperationError()
        return bool(affected)

    def status(self, user_id: str, session_id: str) -> dict[str, object] | None:
        if self.client is None:
            return None
        ok, row = self.client.execute_query(
            """
            SELECT turn_id, fencing_token, stop_requested, lease_until, updated_at
            FROM cs_turn_leases
            WHERE user_id = %s AND session_id = %s
              AND lease_until > UTC_TIMESTAMP(6)
            """,
            (user_id, session_id),
            fetch_one=True,
        )
        if not ok:
            raise StorageOperationError()
        if row is None:
            return None
        return {
            "session_id": session_id,
            "turn_id": str(row["turn_id"]),
            "status": "running",
            "running": True,
            "stop_requested": bool(row["stop_requested"]),
            "fencing_token": int(row["fencing_token"]),
            "lease_until": _serialize_time(row["lease_until"]),
            "updated_at": _serialize_time(row["updated_at"]),
            "steps": [],
        }

    async def _heartbeat(self, lease: TurnLease, owner_task: asyncio.Task | None) -> None:
        interval = get_settings().turn_lease_heartbeat_seconds
        while True:
            await asyncio.sleep(interval)
            state = await asyncio.to_thread(self.renew, lease)
            if state == "active":
                continue
            lease.stop_requested.set()
            if owner_task is not None and not owner_task.done():
                owner_task.cancel()
            return

    def _require_client(self) -> "MySQLClient":
        if self.client is None:
            raise StorageUnavailableError()
        return self.client


_manager = TurnLeaseManager(None)


def initialize_turn_lease_manager(client: "MySQLClient | None") -> None:
    global _manager
    _manager = TurnLeaseManager(client)


def get_turn_lease_manager() -> TurnLeaseManager:
    return _manager


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_utc_naive(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed


def _serialize_time(value: datetime | str) -> str:
    if isinstance(value, datetime):
        resolved = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return resolved.isoformat()
    return str(value)
