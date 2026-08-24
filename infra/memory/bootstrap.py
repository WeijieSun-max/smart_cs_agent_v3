from __future__ import annotations

from domain.customer_service_agent.service import conversation_archive_service, memory_service, short_term_memory_service
from infra.cache import redis_client
from infra.db import mysql_client
from infra.memory.mysql_conversation_archive import MySQLConversationArchive
from infra.memory.mysql_memory_repository import MySQLMemoryRepository
from infra.memory.openai_memory_embedder import OpenAIMemoryEmbedder
from infra.memory.qdrant_memory_vector_index import DisabledMemoryVectorIndex, QdrantMemoryVectorIndex
from infra.memory.persistent_conversation_memory import PersistentConversationMemory
from infra.memory.short_term_memory import RedisShortTermMemory
from pkg.config.settings import get_settings
from pkg.log.logger import get_logger
from pkg.telemetry import normalize_error

logger = get_logger()


def initialize_memory() -> None:
    settings = get_settings()
    redis_client.initialize_redis_client()
    mysql_client.initialize_mysql_client()
    database = mysql_client.get_mysql_client()
    cache = RedisShortTermMemory(redis_client.get_redis_client(), ttl_seconds=settings.redis_cache_expire)
    repository = MySQLMemoryRepository(database) if settings.memory_layered_enabled else None
    archive = MySQLConversationArchive(database, memory_repository=repository)
    conversation_archive_service.initialize_service(archive)
    memory = PersistentConversationMemory(cache, archive)
    short_term_memory_service.initialize_service(memory)
    if settings.memory_layered_enabled:
        assert repository is not None
        vector_index = DisabledMemoryVectorIndex()
        embedder = None
        if settings.memory_qdrant_enabled:
            try:
                from langchain_openai import OpenAIEmbeddings
                from qdrant_client import QdrantClient

                provider = OpenAIEmbeddings(
                    openai_api_base=settings.embedding_base_url,
                    openai_api_key=settings.embedding_api_key,
                    model=settings.embedding_model,
                    request_timeout=settings.embedding_timeout_seconds,
                    max_retries=1,
                    check_embedding_ctx_length=False,
                )
                embedder = OpenAIMemoryEmbedder(provider, settings.embedding_dim)
                vector_index = QdrantMemoryVectorIndex(
                    QdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key,
                        timeout=settings.qdrant_timeout_seconds,
                    ),
                    settings.qdrant_collection,
                    vector_size=settings.embedding_dim,
                    timeout_seconds=settings.qdrant_timeout_seconds,
                )
                vector_index.initialize()
            except Exception as exc:
                error = normalize_error(exc)
                logger.warning(
                    "Qdrant memory index unavailable error_type={} error_code={}",
                    error["error_type"],
                    error["error_code"],
                )
                vector_index = DisabledMemoryVectorIndex()
                embedder = None
        memory_service.initialize_service(repository, vector_index=vector_index, embedder=embedder)
    migrated = memory.migrate_cache_to_archive()
    logger.info("Memory module initialized with Redis cache and MySQL archive migrated_sessions={}", migrated)
