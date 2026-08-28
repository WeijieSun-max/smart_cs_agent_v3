"""长期记忆仓储、向量索引和嵌入器的领域依赖集合。"""

from __future__ import annotations

from typing import Optional

from domain.customer_service_agent.interfaces.i_memory_repository import IMemoryRepository


class MemoryService:
    """集中暴露记忆事实源与可选的可重建语义组件。"""

    def __init__(self, repository: IMemoryRepository, vector_index=None, embedder=None) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.embedder = embedder


instance: Optional[MemoryService] = None


def initialize_service(repository: IMemoryRepository, vector_index=None, embedder=None) -> None:
    """安装进程级记忆依赖集合。"""

    global instance
    instance = MemoryService(repository, vector_index=vector_index, embedder=embedder)


def get_service() -> MemoryService:
    """取得记忆服务；未初始化时立即失败。"""

    if instance is None:
        raise RuntimeError("Memory service is not initialized")
    return instance


def get_service_or_none() -> MemoryService | None:
    """供记忆功能开关或健康检查安全探测服务。"""

    return instance
