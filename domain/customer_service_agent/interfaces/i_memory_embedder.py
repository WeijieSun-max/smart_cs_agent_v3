from __future__ import annotations

from abc import ABC, abstractmethod


class IMemoryEmbedder(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass
