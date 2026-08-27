from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from adapter.web.routes import memory_controller
from domain.customer_service_agent.memory.models import MemoryItem, MemoryStatus, MemoryType
from domain.customer_service_agent.service import memory_service
from pkg.security import get_local_user_id


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


def _item(*, user_id: str | None = None, memory_type=MemoryType.PREFERENCE) -> MemoryItem:
    return MemoryItem(
        user_id=user_id or get_local_user_id(),
        memory_type=memory_type,
        memory_key="contact.channel",
        content="偏好短信联系",
        confidence=0.95,
        status=MemoryStatus.ACTIVE,
        version=1,
        expires_at=NOW + timedelta(days=30),
        created_at=NOW,
        updated_at=NOW,
    )


class FakeRepository:
    available = True

    def __init__(self):
        self.items = [_item(), _item(memory_type=MemoryType.TASK)]
        self.last_user_id = None
        self.last_filters = None

    def list_items(self, user_id, *, memory_type=None, status=None, cursor=None, limit=50):
        self.last_user_id = user_id
        self.last_filters = (memory_type, status, cursor, limit)
        items = [item for item in self.items if item.user_id == user_id]
        if memory_type:
            items = [item for item in items if item.memory_type == memory_type]
        if status:
            items = [item for item in items if item.status == status]
        return items[:limit], "next-page" if len(items) > limit else None

    def correct_item(self, user_id, memory_id, *, content, structured_data, reason, actor):
        self.last_user_id = user_id
        for item in self.items:
            if str(item.memory_id) == memory_id and item.user_id == user_id:
                corrected = item.model_copy(update={
                    "memory_id": uuid4(),
                    "content": content,
                    "structured_data": structured_data or item.structured_data,
                    "version": item.version + 1,
                    "supersedes_id": item.memory_id,
                    "confidence": 1.0,
                    "updated_at": NOW + timedelta(seconds=1),
                    "created_at": NOW + timedelta(seconds=1),
                })
                item.status = MemoryStatus.SUPERSEDED
                self.items.append(corrected)
                return corrected
        return None

    def hard_delete_item(self, user_id, memory_id, *, reason, actor):
        self.last_user_id = user_id
        original = len(self.items)
        self.items = [
            item for item in self.items
            if not (str(item.memory_id) == memory_id and item.user_id == user_id)
        ]
        return len(self.items) != original

    def purge_items(self, user_id, memory_type, *, reason, actor):
        self.last_user_id = user_id
        original = len(self.items)
        self.items = [
            item for item in self.items
            if item.user_id != user_id or (memory_type is not None and item.memory_type != memory_type)
        ]
        return original - len(self.items)


def _client(repository: FakeRepository) -> TestClient:
    memory_service.initialize_service(repository)
    app = FastAPI()
    app.include_router(memory_controller.router)
    return TestClient(app)


def test_list_memories_supports_filters_and_cursor_without_accepting_user_identity() -> None:
    repository = FakeRepository()
    client = _client(repository)

    response = client.get(
        "/api/memories",
        params={"type": "preference", "status": "active", "limit": 1, "user_id": "attacker"},
    )

    assert response.status_code == 200
    assert len(response.json()["items"]) == 1
    assert repository.last_user_id == get_local_user_id()
    assert repository.last_filters == (MemoryType.PREFERENCE, MemoryStatus.ACTIVE, None, 1)


def test_correction_creates_a_new_version_and_rejects_spoofed_user_in_body() -> None:
    repository = FakeRepository()
    client = _client(repository)
    old = repository.items[0]

    spoofed = client.patch(
        f"/api/memories/{old.memory_id}",
        json={"content": "改为电话联系", "user_id": "attacker"},
    )
    corrected = client.patch(
        f"/api/memories/{old.memory_id}",
        json={"content": "改为电话联系", "reason": "用户明确纠正"},
    )

    assert spoofed.status_code == 422
    assert corrected.status_code == 200
    payload = corrected.json()
    assert payload["version"] == 2
    assert payload["supersedes_id"] == str(old.memory_id)
    assert payload["content"] == "改为电话联系"
    assert repository.last_user_id == get_local_user_id()


def test_forget_is_idempotently_not_found_after_hard_delete() -> None:
    repository = FakeRepository()
    client = _client(repository)
    memory_id = repository.items[0].memory_id

    assert client.delete(f"/api/memories/{memory_id}").status_code == 204
    assert client.delete(f"/api/memories/{memory_id}").status_code == 404


def test_purge_requires_explicit_confirmation_and_can_clear_one_type() -> None:
    repository = FakeRepository()
    client = _client(repository)

    rejected = client.post("/api/memories/purge", json={"memory_type": "preference"})
    accepted = client.post(
        "/api/memories/purge",
        json={"confirmation": "PURGE", "memory_type": "preference"},
    )

    assert rejected.status_code == 422
    assert accepted.status_code == 200
    assert accepted.json() == {"deleted_count": 1}
    assert [item.memory_type for item in repository.items] == [MemoryType.TASK]


def test_memory_id_path_rejects_non_uuid_values() -> None:
    response = _client(FakeRepository()).delete("/api/memories/not-a-uuid")

    assert response.status_code == 422
