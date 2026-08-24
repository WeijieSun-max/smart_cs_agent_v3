from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class ITicketRepository(ABC):
    @abstractmethod
    def create(
        self,
        ticket_type: str,
        priority: str,
        summary: str,
        details: str,
        user_id: str,
        operation_key: str | None = None,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def query(self, ticket_id: str) -> Optional[dict[str, Any]]:
        pass

    @abstractmethod
    def update_status(self, ticket_id: str, status: str) -> Optional[dict[str, Any]]:
        pass
