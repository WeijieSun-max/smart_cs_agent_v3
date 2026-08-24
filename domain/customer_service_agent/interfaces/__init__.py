from __future__ import annotations

from domain.customer_service_agent.interfaces.i_conversation_archive import IConversationArchive
from domain.customer_service_agent.interfaces.i_knowledge_store import IKnowledgeStore
from domain.customer_service_agent.interfaces.i_memory_embedder import IMemoryEmbedder
from domain.customer_service_agent.interfaces.i_memory_repository import IMemoryRepository
from domain.customer_service_agent.interfaces.i_memory_vector_index import IMemoryVectorIndex
from domain.customer_service_agent.interfaces.i_short_term_memory import IShortTermMemory

__all__ = [
    "IConversationArchive",
    "IKnowledgeStore",
    "IMemoryEmbedder",
    "IMemoryRepository",
    "IMemoryVectorIndex",
    "IShortTermMemory",
]
