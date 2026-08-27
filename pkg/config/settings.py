from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field("smart-cs-fc-agent", alias="APP_NAME")
    debug: bool = Field(False, alias="DEBUG")
    server_host: str = Field("127.0.0.1", alias="SERVER_HOST")
    server_port: int = Field(8000, alias="SERVER_PORT")
    local_user_id: str = Field(
        "local-user",
        alias="LOCAL_USER_ID",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    chat_message_max_length: int = Field(8000, alias="CHAT_MESSAGE_MAX_LENGTH", ge=1, le=100_000)

    qwen_api_key: str = Field("", alias="QWEN_API_KEY", repr=False)
    qwen_base_url: str = Field(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="QWEN_BASE_URL",
    )
    qwen_model: str = Field("qwen3.7-flash", alias="QWEN_MODEL")
    qwen_supervisor_model: str = Field("", alias="QWEN_SUPERVISOR_MODEL")
    qwen_supervisor_base_url: str = Field("", alias="QWEN_SUPERVISOR_BASE_URL")
    qwen_knowledge_agent_model: str = Field("", alias="QWEN_KNOWLEDGE_AGENT_MODEL")
    qwen_knowledge_agent_base_url: str = Field("", alias="QWEN_KNOWLEDGE_AGENT_BASE_URL")
    qwen_telecom_agent_model: str = Field("", alias="QWEN_TELECOM_AGENT_MODEL")
    qwen_telecom_agent_base_url: str = Field("", alias="QWEN_TELECOM_AGENT_BASE_URL")
    qwen_retail_agent_model: str = Field("", alias="QWEN_RETAIL_AGENT_MODEL")
    qwen_retail_agent_base_url: str = Field("", alias="QWEN_RETAIL_AGENT_BASE_URL")
    qwen_response_writer_model: str = Field("", alias="QWEN_RESPONSE_WRITER_MODEL")
    qwen_response_writer_base_url: str = Field("", alias="QWEN_RESPONSE_WRITER_BASE_URL")
    qwen_safety_guard_model: str = Field("", alias="QWEN_SAFETY_GUARD_MODEL")
    qwen_safety_guard_base_url: str = Field("", alias="QWEN_SAFETY_GUARD_BASE_URL")
    qwen_memory_model: str = Field("", alias="QWEN_MEMORY_MODEL")
    qwen_memory_base_url: str = Field("", alias="QWEN_MEMORY_BASE_URL")
    llm_max_concurrency: int = Field(20, alias="LLM_MAX_CONCURRENCY", ge=1, le=200)
    llm_queue_capacity: int = Field(40, alias="LLM_QUEUE_CAPACITY", ge=1, le=1000)
    llm_queue_timeout_seconds: float = Field(15.0, alias="LLM_QUEUE_TIMEOUT_SECONDS", ge=0.1, le=120)

    db_host: str = Field("localhost", alias="DB_HOST")
    db_port: int = Field(3306, alias="DB_PORT")
    db_user: str = Field("root", alias="DB_USER")
    db_password: str = Field("", alias="DB_PASSWORD")
    db_name: str = Field("assist_gen", alias="DB_NAME")
    db_pool_max_connections: int = Field(10, alias="DB_POOL_MAX_CONNECTIONS", ge=1)
    db_pool_min_cached: int = Field(1, alias="DB_POOL_MIN_CACHED", ge=0)
    db_pool_max_cached: int = Field(5, alias="DB_POOL_MAX_CACHED", ge=0)
    db_pool_blocking: bool = Field(False, alias="DB_POOL_BLOCKING")
    db_connect_timeout_seconds: int = Field(5, alias="DB_CONNECT_TIMEOUT_SECONDS", ge=1, le=120)
    db_read_timeout_seconds: int = Field(10, alias="DB_READ_TIMEOUT_SECONDS", ge=1, le=300)
    db_write_timeout_seconds: int = Field(10, alias="DB_WRITE_TIMEOUT_SECONDS", ge=1, le=300)
    pii_encryption_key: str = Field("", alias="PII_ENCRYPTION_KEY", repr=False)

    redis_host: str = Field("localhost", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")
    redis_db: int = Field(0, alias="REDIS_DB")
    redis_password: str = Field("", alias="REDIS_PASSWORD")
    redis_cache_expire: int = Field(3600, alias="REDIS_CACHE_EXPIRE")
    redis_enabled: bool = Field(True, alias="REDIS_ENABLED")
    redis_max_connections: int = Field(20, alias="REDIS_MAX_CONNECTIONS", ge=1)
    redis_health_check_interval: int = Field(30, alias="REDIS_HEALTH_CHECK_INTERVAL", ge=0)

    vector_store_path: str = Field("data/vector_store/faiss_index", alias="VECTOR_STORE_PATH")
    embedding_dim: int = Field(1536, alias="EMBEDDING_DIM")
    embedding_base_url: str = Field("", alias="EMBEDDING_BASE_URL")
    embedding_api_key: str = Field("", alias="EMBEDDING_API_KEY")
    embedding_model: str = Field("", alias="EMBEDDING_MODEL")
    embedding_timeout_seconds: int = Field(10, alias="EMBEDDING_TIMEOUT_SECONDS", ge=1, le=120)
    rag_keyword_min_score: float = Field(0.18, alias="RAG_KEYWORD_MIN_SCORE", ge=0.0, le=1.0)
    rag_candidate_limit: int = Field(8, alias="RAG_CANDIDATE_LIMIT", ge=1, le=100)
    knowledge_qdrant_enabled: bool = Field(False, alias="KNOWLEDGE_QDRANT_ENABLED")
    knowledge_qdrant_collection: str = Field(
        "knowledge_10086_v1",
        alias="KNOWLEDGE_QDRANT_COLLECTION",
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    knowledge_qdrant_timeout_seconds: float = Field(30.0, alias="KNOWLEDGE_QDRANT_TIMEOUT_SECONDS", ge=0.1, le=120.0)
    knowledge_embedding_mode: str = Field("external", alias="KNOWLEDGE_EMBEDDING_MODE")
    knowledge_retrieval_mode: str = Field("hybrid", alias="KNOWLEDGE_RETRIEVAL_MODE")
    knowledge_dense_vector_name: str = Field("dense", alias="KNOWLEDGE_DENSE_VECTOR_NAME", pattern=r"^[A-Za-z0-9._-]+$")
    knowledge_sparse_vector_name: str = Field("sparse", alias="KNOWLEDGE_SPARSE_VECTOR_NAME", pattern=r"^[A-Za-z0-9._-]+$")
    knowledge_sparse_encoder: str = Field("char_bigram", alias="KNOWLEDGE_SPARSE_ENCODER")
    knowledge_qdrant_dense_model: str = Field("", alias="KNOWLEDGE_QDRANT_DENSE_MODEL")
    knowledge_qdrant_sparse_model: str = Field("Qdrant/bm25", alias="KNOWLEDGE_QDRANT_SPARSE_MODEL")
    knowledge_fusion: str = Field("rrf", alias="KNOWLEDGE_FUSION")
    knowledge_dense_weight: float = Field(0.7, alias="KNOWLEDGE_DENSE_WEIGHT", ge=0.0, le=1.0)
    knowledge_sparse_weight: float = Field(0.3, alias="KNOWLEDGE_SPARSE_WEIGHT", ge=0.0, le=1.0)
    knowledge_prefetch_limit: int = Field(40, alias="KNOWLEDGE_PREFETCH_LIMIT", ge=1, le=1000)
    knowledge_dense_score_threshold: float | None = Field(None, alias="KNOWLEDGE_DENSE_SCORE_THRESHOLD", ge=0.0, le=1.0)

    memory_layered_enabled: bool = Field(False, alias="MEMORY_LAYERED_ENABLED")
    memory_qdrant_enabled: bool = Field(False, alias="MEMORY_QDRANT_ENABLED")
    qdrant_url: str = Field("", alias="QDRANT_URL", max_length=2048)
    qdrant_api_key: str = Field("", alias="QDRANT_API_KEY", repr=False, max_length=4096)
    qdrant_collection: str = Field(
        "customer_memory_v1",
        alias="QDRANT_COLLECTION",
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    qdrant_timeout_seconds: float = Field(0.8, alias="QDRANT_TIMEOUT_SECONDS", ge=0.1, le=30.0)

    memory_context_max_tokens: int = Field(1800, alias="MEMORY_CONTEXT_MAX_TOKENS", ge=100, le=32_000)
    memory_recent_messages_tokens: int = Field(700, alias="MEMORY_RECENT_MESSAGES_TOKENS", ge=0, le=32_000)
    memory_session_summary_tokens: int = Field(350, alias="MEMORY_SESSION_SUMMARY_TOKENS", ge=0, le=32_000)
    memory_episode_tokens: int = Field(350, alias="MEMORY_EPISODE_TOKENS", ge=0, le=32_000)
    memory_semantic_tokens: int = Field(400, alias="MEMORY_SEMANTIC_TOKENS", ge=0, le=32_000)
    memory_recall_top_k: int = Field(20, alias="MEMORY_RECALL_TOP_K", ge=1, le=100)
    memory_recall_min_score: float = Field(0.35, alias="MEMORY_RECALL_MIN_SCORE", ge=0.0, le=1.0)
    memory_fallback_items: int = Field(10, alias="MEMORY_FALLBACK_ITEMS", ge=1, le=100)

    memory_episode_min_confidence: float = Field(0.70, alias="MEMORY_EPISODE_MIN_CONFIDENCE", ge=0.0, le=1.0)
    memory_preference_min_confidence: float = Field(0.85, alias="MEMORY_PREFERENCE_MIN_CONFIDENCE", ge=0.0, le=1.0)
    memory_fact_min_confidence: float = Field(0.85, alias="MEMORY_FACT_MIN_CONFIDENCE", ge=0.0, le=1.0)
    memory_task_min_confidence: float = Field(0.85, alias="MEMORY_TASK_MIN_CONFIDENCE", ge=0.0, le=1.0)
    memory_episode_ttl_days: int = Field(180, alias="MEMORY_EPISODE_TTL_DAYS", ge=1, le=3650)
    memory_preference_ttl_days: int = Field(365, alias="MEMORY_PREFERENCE_TTL_DAYS", ge=1, le=3650)
    memory_fact_ttl_days: int = Field(180, alias="MEMORY_FACT_TTL_DAYS", ge=1, le=3650)
    memory_task_active_ttl_days: int = Field(90, alias="MEMORY_TASK_ACTIVE_TTL_DAYS", ge=1, le=3650)

    memory_worker_enabled: bool = Field(False, alias="MEMORY_WORKER_ENABLED")
    memory_worker_max_attempts: int = Field(5, alias="MEMORY_WORKER_MAX_ATTEMPTS", ge=1, le=20)
    memory_worker_lease_seconds: int = Field(60, alias="MEMORY_WORKER_LEASE_SECONDS", ge=5, le=3600)
    memory_worker_poll_seconds: float = Field(1.0, alias="MEMORY_WORKER_POLL_SECONDS", ge=0.1, le=60.0)

    agent_state_ttl_seconds: int = Field(3600, alias="AGENT_STATE_TTL_SECONDS", ge=60)
    tool_call_log_limit: int = Field(1000, alias="TOOL_CALL_LOG_LIMIT", ge=10, le=100_000)
    skill_root: str = Field("skills", alias="SKILL_ROOT", pattern=r"^[A-Za-z0-9._/-]+$")
    skill_files_enabled: bool = Field(True, alias="SKILL_FILES_ENABLED")
    identity_header_enabled: bool = Field(False, alias="IDENTITY_HEADER_ENABLED")
    trusted_proxy_networks: str = Field("127.0.0.1/32,::1/128", alias="TRUSTED_PROXY_NETWORKS")
    session_lock_timeout_seconds: float = Field(15.0, alias="SESSION_LOCK_TIMEOUT_SECONDS", ge=0.1, le=120)
    per_user_queue_limit: int = Field(2, alias="PER_USER_QUEUE_LIMIT", ge=1, le=20)
    memory_summary_eligible_turns: int = Field(20, alias="MEMORY_SUMMARY_ELIGIBLE_TURNS", ge=2, le=1000)
    memory_summary_increment_turns: int = Field(8, alias="MEMORY_SUMMARY_INCREMENT_TURNS", ge=1, le=100)
    action_reconcile_worker_enabled: bool = Field(False, alias="ACTION_RECONCILE_WORKER_ENABLED")

    skill_execution_timeout_seconds: float = Field(
        30.0,
        alias="SKILL_EXECUTION_TIMEOUT_SECONDS",
        ge=30.0,
        le=30.0,
    )
    pending_action_ttl_seconds: int = Field(
        900,
        alias="PENDING_ACTION_TTL_SECONDS",
        ge=60,
        le=86_400,
    )
    langfuse_enabled: bool = Field(False, alias="LANGFUSE_ENABLED")
    langfuse_public_key: str = Field("", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field("", alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str = Field("https://cloud.langfuse.com", alias="LANGFUSE_BASE_URL")
    langfuse_environment: str = Field("development", alias="LANGFUSE_TRACING_ENVIRONMENT")
    langfuse_release: str = Field("smart-cs-agent-v2", alias="LANGFUSE_RELEASE")
    langfuse_sample_rate: float = Field(1.0, alias="LANGFUSE_SAMPLE_RATE", ge=0.0, le=1.0)
    langfuse_capture_content: bool = Field(False, alias="LANGFUSE_CAPTURE_CONTENT")
    langfuse_hash_salt: str = Field("", alias="LANGFUSE_HASH_SALT")

    @property
    def root_dir(self) -> Path:
        return ROOT_DIR

    @field_validator("knowledge_dense_score_threshold", mode="before")
    @classmethod
    def validate_optional_float(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_cross_field_contracts(self) -> "Settings":
        layer_budget = (
            self.memory_recent_messages_tokens
            + self.memory_session_summary_tokens
            + self.memory_episode_tokens
            + self.memory_semantic_tokens
        )
        if layer_budget > self.memory_context_max_tokens:
            raise ValueError("memory layer budgets cannot exceed MEMORY_CONTEXT_MAX_TOKENS")
        if self.knowledge_embedding_mode not in {"external", "qdrant"}:
            raise ValueError("KNOWLEDGE_EMBEDDING_MODE must be external or qdrant")
        if self.knowledge_retrieval_mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError("KNOWLEDGE_RETRIEVAL_MODE must be dense, sparse, or hybrid")
        if self.knowledge_sparse_encoder not in {"char_bigram", "qdrant_bm25"}:
            raise ValueError("KNOWLEDGE_SPARSE_ENCODER must be char_bigram or qdrant_bm25")
        if self.knowledge_fusion not in {"rrf", "dbsf", "weighted_rrf", "client_weighted_rrf"}:
            raise ValueError("KNOWLEDGE_FUSION must be rrf, dbsf, weighted_rrf, or client_weighted_rrf")
        if self.knowledge_retrieval_mode == "hybrid" and (
            self.knowledge_dense_weight + self.knowledge_sparse_weight <= 0
        ):
            raise ValueError("knowledge hybrid weights must have a positive sum")
        if self.knowledge_qdrant_enabled:
            required = {
                "QDRANT_URL": self.qdrant_url,
                "QDRANT_API_KEY": self.qdrant_api_key,
            }
            if self.knowledge_embedding_mode == "external":
                required.update({
                    "EMBEDDING_BASE_URL": self.embedding_base_url,
                    "EMBEDDING_API_KEY": self.embedding_api_key,
                    "EMBEDDING_MODEL": self.embedding_model,
                })
            if self.knowledge_embedding_mode == "qdrant" and not (
                self.knowledge_qdrant_dense_model or self.embedding_model
            ):
                required["KNOWLEDGE_QDRANT_DENSE_MODEL"] = ""
            missing = [name for name, value in required.items() if not value.strip()]
            if missing:
                raise ValueError(f"Qdrant knowledge requires: {', '.join(missing)}")
        if self.memory_qdrant_enabled:
            if not self.memory_layered_enabled:
                raise ValueError("MEMORY_QDRANT_ENABLED requires MEMORY_LAYERED_ENABLED")
            if not self.qdrant_url.startswith("https://"):
                raise ValueError("QDRANT_URL must use https for Qdrant Cloud")
            required = {
                "QDRANT_API_KEY": self.qdrant_api_key,
                "EMBEDDING_BASE_URL": self.embedding_base_url,
                "EMBEDDING_API_KEY": self.embedding_api_key,
                "EMBEDDING_MODEL": self.embedding_model,
            }
            missing = [name for name, value in required.items() if not value.strip()]
            if missing:
                raise ValueError(f"Qdrant memory requires: {', '.join(missing)}")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
