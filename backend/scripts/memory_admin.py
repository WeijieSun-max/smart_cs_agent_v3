from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from application.customer_service.memory_admin_service import MemoryAdminService  # noqa: E402
from infra.db import mysql_client  # noqa: E402
from infra.memory.mysql_memory_repository import MySQLMemoryRepository  # noqa: E402
from infra.memory.openai_memory_embedder import OpenAIMemoryEmbedder  # noqa: E402
from infra.memory.qdrant_memory_vector_index import QdrantMemoryVectorIndex  # noqa: E402
from pkg.config.settings import get_settings  # noqa: E402


def _uuid(value: str) -> str:
    return str(UUID(value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layered-memory offline administration")
    commands = parser.add_subparsers(dest="command", required=True)
    reindex = commands.add_parser("reindex", help="Rebuild active MySQL memories into a new Qdrant collection")
    reindex.add_argument("--collection", required=True)
    reindex.add_argument("--cursor", type=_uuid)
    reindex.add_argument("--batch-size", type=int, default=100, choices=range(1, 1001), metavar="1..1000")
    reindex.add_argument("--max-batches", type=int, default=0, help="0 processes until completion")
    replay = commands.add_parser("replay-dead-letter", help="Replay exactly one dead outbox event")
    replay.add_argument("--event-id", required=True, type=_uuid)
    return parser


def _repository() -> MySQLMemoryRepository:
    mysql_client.initialize_mysql_client()
    repository = MySQLMemoryRepository(mysql_client.get_mysql_client())
    if not repository.available:
        raise RuntimeError("MySQL memory repository is unavailable")
    return repository


def _reindex(args) -> dict[str, object]:
    settings = get_settings()
    if not settings.memory_qdrant_enabled:
        raise RuntimeError("MEMORY_QDRANT_ENABLED must be true for reindex")
    if args.collection == settings.qdrant_collection:
        raise ValueError("reindex target must be a new collection")
    from langchain_openai import OpenAIEmbeddings
    from qdrant_client import QdrantClient

    vector_index = QdrantMemoryVectorIndex(
        QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout_seconds,
        ),
        args.collection,
        vector_size=settings.embedding_dim,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    vector_index.initialize()
    embedder = OpenAIMemoryEmbedder(
        OpenAIEmbeddings(
            openai_api_base=settings.embedding_base_url,
            openai_api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            request_timeout=settings.embedding_timeout_seconds,
            max_retries=1,
            check_embedding_ctx_length=False,
        ),
        settings.embedding_dim,
    )
    service = MemoryAdminService(_repository(), vector_index, embedder)
    cursor = args.cursor
    total = 0
    batches = 0
    completed = False
    while not completed and (args.max_batches <= 0 or batches < args.max_batches):
        result = service.reindex_batch(cursor=cursor, batch_size=args.batch_size)
        total += result.indexed_count
        batches += 1
        cursor = result.next_cursor
        completed = result.completed
    return {
        "operation": "reindex",
        "collection": args.collection,
        "indexed_count": total,
        "batches": batches,
        "completed": completed,
        "next_cursor": cursor,
    }


def _replay(args) -> dict[str, object]:
    service = MemoryAdminService(_repository(), None, None)
    replayed = service.replay_dead_letter(args.event_id)
    return {"operation": "replay-dead-letter", "event_id": args.event_id, "replayed": replayed}


def main() -> int:
    args = _parser().parse_args()
    result = _reindex(args) if args.command == "reindex" else _replay(args)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    if args.command == "reindex":
        return 0
    return 0 if result.get("replayed", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
