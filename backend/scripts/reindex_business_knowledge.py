from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.knowledge.document_loaders import load_10086_records  # noqa: E402
from infra.knowledge.qdrant_knowledge_store import QdrantKnowledgeStore  # noqa: E402
from pkg.config.settings import get_settings  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild 10086 knowledge records into Qdrant")
    parser.add_argument("--dry-run", action="store_true", help="Parse documents and print counts without embedding or Qdrant writes")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the target Qdrant collection before upsert")
    parser.add_argument("--batch-size", type=int, default=20, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--timeout-seconds", type=float, help="Override KNOWLEDGE_QDRANT_TIMEOUT_SECONDS for this run")
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = get_settings()
    records = load_10086_records(settings.root_dir)
    counts = _counts(records)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", **counts}, ensure_ascii=False, indent=2))
        return
    if not settings.knowledge_qdrant_enabled:
        raise RuntimeError("KNOWLEDGE_QDRANT_ENABLED must be true for reindex")
    store = _store(settings, timeout_seconds=args.timeout_seconds)
    try:
        store.initialize(recreate=args.recreate)
    except Exception as exc:
        raise RuntimeError(
            "Qdrant collection initialization timed out or failed. "
            "Check QDRANT_URL/proxy/network, or retry with --timeout-seconds 60."
        ) from exc
    for batch in _batches(records, args.batch_size):
        store.upsert_records(batch)
    print(json.dumps({"status": "indexed", **counts}, ensure_ascii=False, indent=2))


def _store(settings, *, timeout_seconds: float | None = None) -> QdrantKnowledgeStore:
    from langchain_openai import OpenAIEmbeddings
    from qdrant_client import QdrantClient

    timeout = timeout_seconds or settings.knowledge_qdrant_timeout_seconds
    embedding_provider = None
    if settings.knowledge_embedding_mode == "external":
        embedding_provider = OpenAIEmbeddings(
            openai_api_base=settings.embedding_base_url,
            openai_api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            request_timeout=settings.embedding_timeout_seconds,
            max_retries=1,
            check_embedding_ctx_length=False,
        )
    client_kwargs = {
        "url": settings.qdrant_url,
        "api_key": settings.qdrant_api_key,
        "timeout": timeout,
    }
    if settings.knowledge_embedding_mode == "qdrant":
        client_kwargs["cloud_inference"] = True
    try:
        client = QdrantClient(**client_kwargs)
    except TypeError:
        client_kwargs.pop("cloud_inference", None)
        client = QdrantClient(**client_kwargs)
    return QdrantKnowledgeStore(
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
        timeout_seconds=timeout,
        dense_score_threshold=settings.knowledge_dense_score_threshold,
    )


def _counts(records) -> dict[str, int]:
    qa = sum(1 for record in records if record.payload.get("source_type") == "qa")
    help_docs = sum(1 for record in records if record.payload.get("source_type") == "help")
    return {"total": len(records), "qa": qa, "help": help_docs}


def _batches(records, size: int):
    for index in range(0, len(records), size):
        yield records[index:index + size]


if __name__ == "__main__":
    main()
