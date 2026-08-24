from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain.customer_service_agent.memory.models import (  # noqa: E402
    MemoryItem,
    MemoryStatus,
    MemoryType,
    VectorSearchHit,
)
from domain.customer_service_agent.service.memory_orchestrator import MemoryOrchestrator  # noqa: E402


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


class _Repository:
    available = True

    def __init__(self, items):
        self.items = items

    def get_latest_summary(self, user_id, session_id):
        return None

    def get_items_by_ids(self, user_id, memory_ids, now):
        wanted = set(memory_ids)
        return [item for item in self.items if item.user_id == user_id and str(item.memory_id) in wanted]

    def list_active_items(self, user_id, **kwargs):
        return [item for item in self.items if item.user_id == user_id]


class _Index:
    available = True

    def __init__(self, hits):
        self.hits = hits

    def search(self, **kwargs):
        return self.hits


class _Embedder:
    def embed_query(self, text):
        return [1.0, 0.0]


class _Recent:
    def get_history(self, session_id):
        return []


def _settings():
    return SimpleNamespace(
        memory_recent_messages_tokens=700,
        memory_session_summary_tokens=350,
        memory_episode_tokens=350,
        memory_semantic_tokens=400,
        memory_context_max_tokens=1800,
        memory_recall_top_k=20,
        memory_recall_min_score=0.35,
        memory_fallback_items=100,
    )


def evaluate(path: Path, *, top_k: int = 5) -> dict[str, object]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    relevant_total = 0
    recalled_total = 0
    latencies: list[float] = []
    case_results = []
    for case_index, case in enumerate(cases, start=1):
        user_id = f"eval-user-{case_index}"
        items = [
            MemoryItem(
                memory_id=memory["id"],
                user_id=user_id,
                memory_type=MemoryType(memory["type"]),
                memory_key=f"eval.{case['name']}.{index}",
                content=memory["content"],
                confidence=0.9,
                status=MemoryStatus.ACTIVE,
                version=1,
                expires_at=NOW + timedelta(days=30),
                created_at=NOW,
                updated_at=NOW,
            )
            for index, memory in enumerate(case["memories"])
        ]
        hits = [
            VectorSearchHit(memory_id=memory["id"], score=memory["score"])
            for memory in case["memories"]
        ]
        orchestrator = MemoryOrchestrator(
            repository=_Repository(items),
            vector_index=_Index(hits),
            embedder=_Embedder(),
            short_term_memory=_Recent(),
            settings=_settings(),
            clock=lambda: NOW,
        )
        started = perf_counter()
        packet = orchestrator.build_packet(user_id, f"eval-session-{case_index}", case["query"], "eval-turn")
        latencies.append((perf_counter() - started) * 1000)
        ranked = [*packet.semantic_memories, *packet.episodes]
        recalled_ids = {str(item.memory_id) for item in ranked[:top_k]}
        relevant_ids = {memory["id"] for memory in case["memories"] if memory["relevant"]}
        recalled = len(relevant_ids & recalled_ids)
        relevant_total += len(relevant_ids)
        recalled_total += recalled
        case_results.append({
            "name": case["name"],
            "relevant": len(relevant_ids),
            "recalled_at_5": recalled,
        })
    sorted_latencies = sorted(latencies)
    p95_index = max(0, min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95 + 0.999) - 1))
    return {
        "case_count": len(cases),
        "recall_at_5": recalled_total / relevant_total if relevant_total else 1.0,
        "p95_ms": sorted_latencies[p95_index] if sorted_latencies else 0.0,
        "cases": case_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate layered-memory retrieval on a fixed judged fixture")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--max-p95-ms", type=float, default=800.0)
    args = parser.parse_args()
    result = evaluate(args.fixture)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result["recall_at_5"] >= args.min_recall and result["p95_ms"] <= args.max_p95_ms else 1


if __name__ == "__main__":
    raise SystemExit(main())
