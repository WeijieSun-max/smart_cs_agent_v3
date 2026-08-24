from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IKnowledgeStore(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def add_document(self, content: str, source: str = "", metadata: dict | None = None) -> str:
        pass
