"""非结构化知识存储的领域门面和健康状态适配。"""

from __future__ import annotations

from typing import Any, Optional

from domain.customer_service_agent.interfaces.i_knowledge_store import IKnowledgeStore
from pkg.telemetry import traced_dependency


class KnowledgeService:
    """统一普通检索、领域过滤检索及可选健康检查。"""

    def __init__(self, store: IKnowledgeStore):
        self.store = store

    @traced_dependency("knowledge.search", "retriever")
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """执行通用知识检索并记录依赖追踪。"""

        return self.store.search(query, top_k=top_k)

    @traced_dependency("knowledge.search_domain", "retriever")
    def search_domain(self, query: str, domain: str, top_k: int = 5, document_type: str | None = None) -> list[dict[str, Any]]:
        """优先使用存储原生过滤，否则在统一结果结构上做领域过滤。"""

        filtered = getattr(self.store, "search_filtered", None)
        if callable(filtered):
            return filtered(query, top_k=top_k, domain=domain, document_type=document_type)
        return [item for item in self.store.search(query, top_k=top_k) if item.get("metadata", {}).get("domain") == domain]

    def add_document(self, content: str, source: str = "", metadata: dict | None = None) -> str:
        """向知识事实源写入文档。"""

        return self.store.add_document(content, source=source, metadata=metadata)

    def health_status(self) -> dict[str, Any]:
        """读取存储健康状态；无专用实现时视为 ready。"""

        health = getattr(self.store, "health_status", None)
        return health() if callable(health) else {"status": "ready"}


instance: Optional[KnowledgeService] = None


def initialize_service(store: IKnowledgeStore) -> None:
    """安装使用指定知识存储的进程级服务。"""

    global instance
    instance = KnowledgeService(store)


def get_service() -> KnowledgeService:
    """返回已初始化知识服务。"""

    if instance is None:
        raise RuntimeError("Knowledge service is not initialized")
    return instance
