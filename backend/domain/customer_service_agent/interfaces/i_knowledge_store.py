"""非结构化业务知识的检索与写入端口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IKnowledgeStore(ABC):
    """隐藏本地或 Qdrant 检索实现，向领域层提供统一结果结构。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """按相关度返回最多 top_k 个带来源信息的知识片段。"""
        pass

    @abstractmethod
    def add_document(self, content: str, source: str = "", metadata: dict | None = None) -> str:
        """写入知识文档并返回稳定文档标识。"""
        pass
