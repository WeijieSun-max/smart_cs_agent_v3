from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain.customer_service_agent.service import knowledge_service  # noqa: E402
from infra.knowledge.bootstrap import initialize_knowledge_store  # noqa: E402
from pkg.config.settings import get_settings  # noqa: E402


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "name": "qa_roaming_types",
        "query": "移动电话漫游分为哪几种？国内漫游采用哪种方式？",
        "expected": {"source_type": "qa", "source_file": "01_账户使用类.md", "question_index": 1},
    },
    {
        "name": "qa_border_landline_charge",
        "query": "客户在边界地不加区号拨打固话怎么收费？",
        "expected": {"source_type": "qa", "source_file": "01_账户使用类.md", "question_index": 2},
    },
    {
        "name": "qa_payment_methods",
        "query": "移动电话有哪些交费方式？推荐哪种交费方式？",
        "expected": {"source_type": "qa", "source_file": "01_账户使用类.md", "question_index": 4},
    },
    {
        "name": "help_user_registration",
        "query": "中国移动邮箱账号怎么注册？注册后怎么激活邮箱？",
        "expected": {"source_type": "help", "source_file": "14_用户注册.md"},
    },
    {
        "name": "help_cash_on_delivery",
        "query": "商城货到付款支持哪些情况？货到付款怎么操作？",
        "expected": {"source_type": "help", "source_file": "29_货到付款.md"},
    },
    {
        "name": "help_return_exchange_policy",
        "query": "中国移动商城退换货原则是什么？哪些商品不能退换？",
        "expected": {"source_type": "help", "source_file": "31_退换货原则.md"},
    },
]


def evaluate(cases: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    settings = get_settings()
    initialize_knowledge_store()
    service = knowledge_service.get_service()
    case_results: list[dict[str, Any]] = []
    latencies: list[float] = []
    reciprocal_ranks: list[float] = []
    hits = 0

    for case in cases:
        started = perf_counter()
        results = service.search(case["query"], top_k=top_k)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        rank = _first_match_rank(results, case["expected"])
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
        case_results.append({
            "name": case["name"],
            "query": case["query"],
            "expected": case["expected"],
            "hit": rank is not None,
            "rank": rank,
            "latency_ms": round(latency_ms, 2),
            "top_results": [_compact_result(item) for item in results],
        })

    return {
        "case_count": len(cases),
        "top_k": top_k,
        "knowledge_qdrant_enabled": settings.knowledge_qdrant_enabled,
        "knowledge_embedding_mode": settings.knowledge_embedding_mode,
        "knowledge_retrieval_mode": settings.knowledge_retrieval_mode,
        "knowledge_fusion": settings.knowledge_fusion,
        "hit_at_k": hits / len(cases) if cases else 1.0,
        "mrr": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 1.0,
        "p95_ms": _p95(latencies),
        "cases": case_results,
    }


def _first_match_rank(results: list[dict[str, Any]], expected: dict[str, Any]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if _matches(result, expected):
            return rank
    return None


def _matches(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    metadata = result.get("metadata", {}) or {}
    for key, value in expected.items():
        actual = result.get(key, metadata.get(key))
        if key == "source_file" and actual is None:
            source = str(result.get("source") or metadata.get("source") or "")
            actual = Path(source).name if source else None
        if actual != value:
            return False
    return True


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata", {}) or {}
    source = str(result.get("source") or metadata.get("source") or "")
    return {
        "id": result.get("id"),
        "score": round(float(result.get("score", 0.0)), 6),
        "source_file": metadata.get("source_file") or (Path(source).name if source else ""),
        "source_type": metadata.get("source_type"),
        "first_level_index": metadata.get("first_level_index"),
        "question_index": metadata.get("question_index"),
        "title": metadata.get("title"),
        "question": metadata.get("question"),
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * 0.95 + 0.999) - 1))
    return round(sorted_values[index], 2)


def _load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_CASES
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("fixture must be a JSON array")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate 10086 knowledge retrieval against expected payload/source hits")
    parser.add_argument("--fixture", type=Path, help="JSON fixture; defaults to built-in 10086 cases")
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 101), metavar="1..100")
    parser.add_argument("--min-hit", type=float, default=0.8)
    parser.add_argument("--min-mrr", type=float, default=0.6)
    parser.add_argument("--max-p95-ms", type=float, default=3000.0)
    parser.add_argument("--summary-only", action="store_true", help="Omit per-result details from output")
    args = parser.parse_args()

    result = evaluate(_load_cases(args.fixture), top_k=args.top_k)
    if args.summary_only:
        result = {key: value for key, value in result.items() if key != "cases"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    passed = (
        result["hit_at_k"] >= args.min_hit
        and result["mrr"] >= args.min_mrr
        and result["p95_ms"] <= args.max_p95_ms
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
