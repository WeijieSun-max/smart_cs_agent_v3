"""可重建的用户长期记忆向量索引端口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from domain.customer_service_agent.memory.models import MemoryType, VectorRecord, VectorSearchHit


class IMemoryVectorIndex(ABC):
    """定义记忆向量的初始化、更新、用户隔离检索和删除。"""

    @property
    @abstractmethod
    def available(self) -> bool:
        """索引是否可用；不可用时上层应回退到权威仓储召回。"""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """校验或创建索引结构。"""
        pass

    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None:
        """按 memory_id 幂等写入向量记录。"""
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
        """在用户、记忆类型和有效时间约束内执行相似度检索。"""
        pass

    @abstractmethod
    def delete(self, memory_ids: list[str]) -> None:
        """批量删除已失效或被遗忘的记忆向量。"""
        pass

    @abstractmethod
    def health_status(self) -> dict[str, object]:
        """返回适合健康检查公开的索引状态。"""
        pass
