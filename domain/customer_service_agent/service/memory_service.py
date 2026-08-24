from __future__ import annotations

from typing import Optional

from domain.customer_service_agent.interfaces.i_memory_repository import IMemoryRepository


class MemoryService:
    def __init__(self, repository: IMemoryRepository, vector_index=None, embedder=None) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.embedder = embedder


instance: Optional[MemoryService] = None


def initialize_service(repository: IMemoryRepository, vector_index=None, embedder=None) -> None:
    global instance
    instance = MemoryService(repository, vector_index=vector_index, embedder=embedder)


def get_service() -> MemoryService:
    if instance is None:
        raise RuntimeError("Memory service is not initialized")
    return instance


def get_service_or_none() -> MemoryService | None:
    return instance
