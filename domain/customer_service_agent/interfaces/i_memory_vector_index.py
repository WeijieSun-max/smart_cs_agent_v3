from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from domain.customer_service_agent.memory.models import MemoryType, VectorRecord, VectorSearchHit


class IMemoryVectorIndex(ABC):
    @property
    @abstractmethod
    def available(self) -> bool:
        pass

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        *,
        user_id: str,
        query_vector: list[float],
        memory_types: list[MemoryType],
        top_k: int,
        now: datetime,
    ) -> list[VectorSearchHit]:
        pass

    @abstractmethod
    def delete(self, memory_ids: list[str]) -> None:
        pass

    @abstractmethod
    def health_status(self) -> dict[str, object]:
        pass
