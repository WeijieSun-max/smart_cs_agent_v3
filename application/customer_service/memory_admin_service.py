from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from domain.customer_service_agent.memory.models import VectorRecord
from domain.customer_service_agent.service import memory_service


@dataclass(frozen=True)
class ReindexBatchResult:
    indexed_count: int
    next_cursor: str | None
    completed: bool


class MemoryAdminService:
    def __init__(
        self,
        repository,
        vector_index,
        embedder,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.embedder = embedder
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def reindex_batch(self, *, cursor: str | None, batch_size: int = 100) -> ReindexBatchResult:
        if batch_size < 1 or batch_size > 1000:
            raise ValueError("batch_size must be between 1 and 1000")
        items, next_cursor = self.repository.scan_active_items(
            cursor=cursor,
            limit=batch_size,
            now=self.clock(),
        )
        if items:
            vectors = self.embedder.embed_documents([item.content for item in items])
            if len(vectors) != len(items):
                raise ValueError("embedding result count mismatch")
            self.vector_index.upsert([
                VectorRecord(
                    memory_id=item.memory_id,
                    vector=vector,
                    user_id=item.user_id,
                    memory_type=item.memory_type,
                    status=item.status,
                    confidence=item.confidence,
                    updated_at=item.updated_at,
                    expires_at=item.expires_at,
                )
                for item, vector in zip(items, vectors)
            ])
        return ReindexBatchResult(
            indexed_count=len(items),
            next_cursor=next_cursor,
            completed=next_cursor is None,
        )

    def replay_dead_letter(self, event_id: str) -> bool:
        return self.repository.replay_dead_letter(event_id, self.clock())

    def outbox_status(self) -> dict[str, int | float | None]:
        return self.repository.outbox_stats(self.clock())


def current_outbox_status() -> dict[str, object]:
    service = memory_service.get_service_or_none()
    if service is None or not service.repository.available:
        return {"status": "unavailable"}
    try:
        return {"status": "ready", **service.repository.outbox_stats(datetime.now(timezone.utc))}
    except Exception:
        return {"status": "unavailable"}
