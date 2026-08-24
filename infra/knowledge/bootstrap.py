from __future__ import annotations

from domain.customer_service_agent.service import knowledge_service
from langchain_openai import OpenAIEmbeddings
from infra.knowledge.local_knowledge_store import LocalKnowledgeStore
from infra.knowledge.qdrant_knowledge_store import QdrantKnowledgeStore
from pkg.config.settings import Settings, get_settings
from pkg.log.logger import get_logger

logger = get_logger()


def initialize_knowledge_store() -> None:
    settings = get_settings()
    embedding_provider = _embedding_provider(settings)
    if settings.knowledge_qdrant_enabled:
        from qdrant_client import QdrantClient

        client_kwargs = {
            "url": settings.qdrant_url,
            "api_key": settings.qdrant_api_key,
            "timeout": settings.knowledge_qdrant_timeout_seconds,
        }
        if settings.knowledge_embedding_mode == "qdrant":
            client_kwargs["cloud_inference"] = True
        try:
            client = QdrantClient(**client_kwargs)
        except TypeError:
            client_kwargs.pop("cloud_inference", None)
            client = QdrantClient(**client_kwargs)
        store = QdrantKnowledgeStore(
            client,
            settings.knowledge_qdrant_collection,
            vector_size=settings.embedding_dim,
            dense_vector_name=settings.knowledge_dense_vector_name,
            sparse_vector_name=settings.knowledge_sparse_vector_name,
            embedding_provider=embedding_provider,
            embedding_mode=settings.knowledge_embedding_mode,
            retrieval_mode=settings.knowledge_retrieval_mode,
            sparse_encoder_name=settings.knowledge_sparse_encoder,
            qdrant_dense_model=settings.knowledge_qdrant_dense_model or settings.embedding_model,
            qdrant_sparse_model=settings.knowledge_qdrant_sparse_model,
            fusion=settings.knowledge_fusion,
            dense_weight=settings.knowledge_dense_weight,
            sparse_weight=settings.knowledge_sparse_weight,
            prefetch_limit=settings.knowledge_prefetch_limit,
            timeout_seconds=settings.knowledge_qdrant_timeout_seconds,
            dense_score_threshold=settings.knowledge_dense_score_threshold,
        )
        store.initialize()
        knowledge_service.initialize_service(store)
        logger.info("Knowledge module initialized with Qdrant collection={}", settings.knowledge_qdrant_collection)
        return

    vector_path = settings.root_dir / settings.vector_store_path
    store = LocalKnowledgeStore(
        index_path=vector_path,
        embedding_dim=settings.embedding_dim,
        embedding_provider=embedding_provider,
        keyword_min_score=settings.rag_keyword_min_score,
        candidate_limit=settings.rag_candidate_limit,
    )
    _seed_knowledge(store)
    knowledge_service.initialize_service(store)
    logger.info("Knowledge module initialized")


def _embedding_provider(settings: Settings) -> OpenAIEmbeddings | None:
    if settings.embedding_base_url and settings.embedding_api_key and settings.embedding_model:
        return OpenAIEmbeddings(
            openai_api_base=settings.embedding_base_url,
            openai_api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            request_timeout=settings.embedding_timeout_seconds,
            max_retries=1,
            check_embedding_ctx_length=False,
        )
    return None


def _seed_knowledge(store: LocalKnowledgeStore) -> None:
    settings = get_settings()
    sources = ((settings.root_dir / "data" / "10086_qa", "telecom", "troubleshooting"), (settings.root_dir / "data" / "10086_help", "retail", "policy"))
    for directory, domain, document_type in sources:
        for path in sorted(directory.glob("*.md")):
            content = path.read_text(encoding="utf-8").strip()
            if content:
                store.add_document(
                    content,
                    source=path.relative_to(settings.root_dir).as_posix(),
                    metadata={"domain":domain,"document_type":document_type,"version":"1.0.0","status":"active"},
                )
    store.add_document(
        "我们的理财产品A年化收益率为3.5%-5.2%，投资期限为6个月至3年，最低投资金额10000元。注意：理财非存款，产品有风险，投资须谨慎。",
        source="product_faq.md",
    )
    store.add_document(
        "退款政策：用户在购买后7天内可申请无理由退款，超过7天需提供合理原因。退款将在3-5个工作日内原路退回。",
        source="refund_policy.md",
    )
    store.add_document(
        "开户流程：1.准备身份证原件 2.填写开户申请表 3.进行视频认证 4.设置交易密码 5.完成风险评估问卷。整个流程约需15-30分钟。",
        source="account_guide.md",
    )
