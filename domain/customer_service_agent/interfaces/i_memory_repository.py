from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from domain.customer_service_agent.memory.models import (
    MemoryItem,
    MemoryOutboxEvent,
    MemoryOutboxEventType,
    MemorySource,
    MemoryStatus,
    MemoryType,
    SessionSummary,
)


class IMemoryRepository(ABC):
    @property
    @abstractmethod
    def available(self) -> bool:
        pass

    @abstractmethod
    def get_latest_summary(self, user_id: str, session_id: str) -> SessionSummary | None:
        pass

    @abstractmethod
    def save_summary(self, summary: SessionSummary) -> None:
        pass

    @abstractmethod
    def get_messages_after(
        self,
        user_id: str,
        session_id: str,
        after_message_id: int,
        limit: int = 1000,
    ) -> list[dict[str, object]]:
        pass

    @abstractmethod
    def get_items_by_ids(self, user_id: str, memory_ids: list[str], now: datetime) -> list[MemoryItem]:
        pass

    @abstractmethod
    def list_active_items(
        self,
        user_id: str,
        *,
        memory_types: list[MemoryType] | None = None,
        limit: int = 20,
        now: datetime,
    ) -> list[MemoryItem]:
        pass

    @abstractmethod
    def scan_active_items(
        self,
        *,
        cursor: str | None,
        limit: int,
        now: datetime,
    ) -> tuple[list[MemoryItem], str | None]:
        pass

    @abstractmethod
    def find_active_by_key(self, user_id: str, memory_type: MemoryType, memory_key: str) -> MemoryItem | None:
        pass

    @abstractmethod
    def save_item(self, item: MemoryItem, sources: list[MemorySource]) -> None:
        pass

    @abstractmethod
    def apply_extraction(
        self,
        summary: SessionSummary,
        items: list[MemoryItem],
        sources: list[MemorySource],
        superseded_ids: list[str],
    ) -> None:
        pass

    @abstractmethod
    def transition_item(self, user_id: str, memory_id: str, status: MemoryStatus, reason: str) -> bool:
        pass

    @abstractmethod
    def list_items(
        self,
        user_id: str,
        *,
        memory_type: MemoryType | None = None,
        status: MemoryStatus | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[MemoryItem], str | None]:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def hard_delete_item(self, user_id: str, memory_id: str, *, reason: str, actor: str) -> bool:
        pass

    @abstractmethod
    def purge_items(self, user_id: str, memory_type: MemoryType | None, *, reason: str, actor: str) -> int:
        pass

    @abstractmethod
    def delete_session_sources(self, user_id: str, session_id: str) -> list[str]:
        """Return memory IDs that became source-less and were deleted."""
        pass

    @abstractmethod
    def enqueue_outbox(self, event: MemoryOutboxEvent) -> bool:
        pass

    @abstractmethod
    def claim_outbox(
        self,
        event_types: list[MemoryOutboxEventType],
        *,
        worker_id: str,
        limit: int,
        lease_seconds: int,
        now: datetime,
    ) -> list[MemoryOutboxEvent]:
        pass

    @abstractmethod
    def complete_outbox(self, event_id: str, processed_at: datetime) -> None:
        pass

    @abstractmethod
    def retry_outbox(
        self,
        event_id: str,
        *,
        error_code: str,
        available_at: datetime,
        max_attempts: int,
    ) -> None:
        pass

    @abstractmethod
    def outbox_stats(self, now: datetime) -> dict[str, int | float | None]:
        pass

    @abstractmethod
    def replay_dead_letter(self, event_id: str, now: datetime) -> bool:
        pass
