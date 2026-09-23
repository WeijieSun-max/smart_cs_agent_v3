from __future__ import annotations

import math

from domain.customer_service_agent.interfaces.i_memory_embedder import IMemoryEmbedder


class OpenAIMemoryEmbedder(IMemoryEmbedder):
    def __init__(self, provider, vector_size: int) -> None:
        self.provider = provider
        self.vector_size = vector_size

    def embed_query(self, text: str) -> list[float]:
        return self._validate(self.provider.embed_query(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.provider.embed_documents(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedding response size mismatch")
        return [self._validate(vector) for vector in vectors]

    def _validate(self, vector) -> list[float]:
        values = [float(value) for value in vector]
        if len(values) != self.vector_size:
            raise ValueError("embedding vector dimension mismatch")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embedding contains non-finite values")
        return values
