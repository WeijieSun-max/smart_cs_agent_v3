"""长期记忆语义向量化端口。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class IMemoryEmbedder(ABC):
    """保证查询和文档使用相同向量空间的嵌入协议。"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """为单个召回查询生成向量。"""
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """按输入顺序批量生成待索引记忆向量。"""
        pass
