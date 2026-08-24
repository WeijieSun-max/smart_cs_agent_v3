from __future__ import annotations

from typing import Any, Optional

from domain.customer_service_agent.interfaces.i_knowledge_store import IKnowledgeStore
from pkg.telemetry import traced_dependency


class KnowledgeService:
    def __init__(self, store: IKnowledgeStore):
        self.store = store

    @traced_dependency("knowledge.search", "retriever")
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.store.search(query, top_k=top_k)

    @traced_dependency("knowledge.search_domain", "retriever")
    def search_domain(self, query: str, domain: str, top_k: int = 5, document_type: str | None = None) -> list[dict[str, Any]]:
        filtered = getattr(self.store, "search_filtered", None)
        if callable(filtered):
            return filtered(query, top_k=top_k, domain=domain, document_type=document_type)
        return [item for item in self.store.search(query, top_k=top_k) if item.get("metadata", {}).get("domain") == domain]

    def add_document(self, content: str, source: str = "", metadata: dict | None = None) -> str:
        return self.store.add_document(content, source=source, metadata=metadata)

    def health_status(self) -> dict[str, Any]:
        health = getattr(self.store, "health_status", None)
        return health() if callable(health) else {"status": "ready"}


instance: Optional[KnowledgeService] = None


def initialize_service(store: IKnowledgeStore) -> None:
    global instance
    instance = KnowledgeService(store)


def get_service() -> KnowledgeService:
    if instance is None:
        raise RuntimeError("Knowledge service is not initialized")
    return instance
