from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from application.customer_service import turn_lease_service


class LeaseCursor:
    def __init__(self, client) -> None:
        self.client = client
        self.result = None

    def execute(self, sql, args=None):
        normalized = " ".join(sql.split())
        row = self.client.row
        if normalized.startswith("SELECT turn_id, owner_id, fencing_token, lease_until"):
            self.result = dict(row) if row is not None else None
            return 1 if row is not None else 0
        if normalized.startswith("INSERT INTO cs_turn_leases"):
            user_id, session_id, turn_id, owner_id, token, lease_until, updated_at = args
            self.client.row = {
                "user_id": user_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "owner_id": owner_id,
                "fencing_token": token,
                "lease_until": lease_until,
                "stop_requested": 0,
                "updated_at": updated_at,
            }
            return 1
        if "SET turn_id = %s" in normalized:
            turn_id, owner_id, token, lease_until, updated_at, user_id, session_id = args
            assert row["user_id"] == user_id and row["session_id"] == session_id
            row.update(
                turn_id=turn_id,
                owner_id=owner_id,
                fencing_token=token,
                lease_until=lease_until,
                stop_requested=0,
                updated_at=updated_at,
            )
            return 1
        if "SET lease_until = %s" in normalized:
            lease_until, updated_at, user_id, session_id, turn_id, owner_id, token = args
            matches = row is not None and (
                row["user_id"],
                row["session_id"],
                row["turn_id"],
                row["owner_id"],
                row["fencing_token"],
            ) == (user_id, session_id, turn_id, owner_id, token)
            if matches:
                row.update(lease_until=lease_until, updated_at=updated_at)
                return 1
            return 0
        if normalized.startswith("SELECT stop_requested"):
            self.result = {"stop_requested": row["stop_requested"]} if row is not None else None
            return 1 if row is not None else 0
        raise AssertionError(normalized)

    def fetchone(self):
        return self.result


class LeaseClient:
    def __init__(self) -> None:
        self.row = None

    def execute_in_transaction(self, operation):
        return True, operation(LeaseCursor(self))

    def execute_update(self, sql, args=None):
        normalized = " ".join(sql.split())
        row = self.row
        if "SET stop_requested = 1" in normalized:
            if row is None or (row["user_id"], row["session_id"]) != tuple(args):
                return True, 0
            row["stop_requested"] = 1
            return True, 1
        if "SET lease_until = %s" in normalized:
            now, updated_at, user_id, session_id, turn_id, owner_id, token = args
            matches = row is not None and (
                row["user_id"],
                row["session_id"],
                row["turn_id"],
                row["owner_id"],
                row["fencing_token"],
            ) == (user_id, session_id, turn_id, owner_id, token)
            if matches:
                row.update(lease_until=now, updated_at=updated_at)
                return True, 1
            return True, 0
        raise AssertionError(normalized)


def _settings():
    return SimpleNamespace(
        turn_lease_seconds=30,
        turn_lease_heartbeat_seconds=5,
        turn_distributed_lease_enabled=True,
    )


def test_expired_lease_increments_fence_and_rejects_stale_owner(monkeypatch) -> None:
    monkeypatch.setattr(turn_lease_service, "get_settings", _settings)
    client = LeaseClient()
    first_manager = turn_lease_service.TurnLeaseManager(client, owner_id="worker-1")
    second_manager = turn_lease_service.TurnLeaseManager(client, owner_id="worker-2")

    first = first_manager.acquire("user-1", "session-1", "turn-1")
    with pytest.raises(RuntimeError, match="already has a running agent"):
        second_manager.acquire("user-1", "session-1", "turn-2")

    client.row["lease_until"] = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)
    second = second_manager.acquire("user-1", "session-1", "turn-2")

    assert first.fencing_token == 1
    assert second.fencing_token == 2
    assert first_manager.renew(first) == "lost"
    assert second_manager.renew(second) == "active"


def test_stop_request_is_visible_to_lease_owner(monkeypatch) -> None:
    monkeypatch.setattr(turn_lease_service, "get_settings", _settings)
    client = LeaseClient()
    manager = turn_lease_service.TurnLeaseManager(client, owner_id="worker-1")
    lease = manager.acquire("user-1", "session-1", "turn-1")

    assert manager.request_stop("user-1", "session-1") is True
    assert manager.renew(lease) == "stop"
