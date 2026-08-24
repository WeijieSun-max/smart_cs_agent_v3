from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from domain.customer_service_agent.interfaces.i_knowledge_store import IKnowledgeStore
from pkg.exceptions.exception import ToolValidationError
from pkg.telemetry import record_fallback


class LocalKnowledgeStore(IKnowledgeStore):
    """Chinese-friendly keyword recall with optional embedding reranking."""

    def __init__(
        self,
        index_path: Path,
        embedding_dim: int = 1536,
        *,
        embedding_provider: Any | None = None,
        keyword_min_score: float = 0.18,
        candidate_limit: int = 8,
    ):
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.embedding_provider = embedding_provider
        self.keyword_min_score = keyword_min_score
        self.candidate_limit = candidate_limit
        self._documents: list[dict[str, Any]] = []
        self._document_embeddings: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        self._embedding_status = "degraded" if embedding_provider is None else "configured"
        self._load_metadata()

    @property
    def embedding_status(self) -> str:
        return self._embedding_status

    def health_status(self) -> dict[str, Any]:
        return {
            "status": self._embedding_status,
            "mode": "hybrid" if self.embedding_provider is not None else "keyword-only",
            "document_count": len(self._documents),
        }

    def _load_metadata(self) -> None:
        metadata_path = self.index_path.with_suffix(".meta.json")
        if not metadata_path.exists():
            return
        try:
            documents = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return
        if isinstance(documents, list):
            self._documents = [item for item in documents if isinstance(item, dict) and item.get("id")]

    def add_document(self, content: str, source: str = "", metadata: dict | None = None) -> str:
        if not content.strip():
            raise ToolValidationError()
        doc_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
        with self._lock:
            if any(doc["id"] == doc_id for doc in self._documents):
                return doc_id
            self._documents.append({
                "id": doc_id,
                "content": content,
                "source": source,
                "metadata": metadata or {},
            })
        return doc_id

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.search_filtered(query, top_k=top_k)

    def search_filtered(self, query: str, top_k: int = 5, *, domain: str | None = None, document_type: str | None = None) -> list[dict[str, Any]]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
            raise ToolValidationError()
        if not query.strip() or len(query) > 4000:
            raise ToolValidationError()
        with self._lock:
            documents = [
                dict(document) for document in self._documents
                if (domain is None or document.get("metadata", {}).get("domain") == domain)
                and (document_type is None or document.get("metadata", {}).get("document_type") == document_type)
                and document.get("metadata", {}).get("status", "active") == "active"
            ]
        if not documents:
            return []

        candidates = self._keyword_recall(query, documents)
        if not candidates:
            return []
        candidates = candidates[:max(top_k, self.candidate_limit)]
        if self.embedding_provider is not None:
            try:
                candidates = self._embedding_rerank(query, candidates)
                self._embedding_status = "ready"
            except Exception:
                self._embedding_status = "degraded"
                record_fallback("embedding")
        return candidates[:top_k]

    def _keyword_recall(self, query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_terms = _terms(query)
        if not query_terms:
            return []
        document_terms = [_terms(str(document.get("content", ""))) for document in documents]
        document_frequency = Counter(
            term
            for terms in document_terms
            for term in set(terms)
        )
        total_documents = len(documents)
        query_weights = {
            term: _term_weight(term) * _idf(total_documents, document_frequency.get(term, 0))
            for term in set(query_terms)
        }
        denominator = sum(query_weights.values()) or 1.0
        results: list[dict[str, Any]] = []
        for document, terms in zip(documents, document_terms):
            term_counts = Counter(terms)
            matched = sum(
                weight * min(1.0, term_counts.get(term, 0))
                for term, weight in query_weights.items()
            )
            score = matched / denominator
            if score < self.keyword_min_score:
                continue
            item = dict(document)
            item["keyword_score"] = float(score)
            item["score"] = float(score)
            results.append(item)
        return sorted(results, key=lambda item: item["keyword_score"], reverse=True)

    def _embedding_rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_vector = np.asarray(self.embedding_provider.embed_query(query), dtype=np.float32)
        missing = [candidate for candidate in candidates if candidate["id"] not in self._document_embeddings]
        if missing:
            vectors = self.embedding_provider.embed_documents([str(item["content"]) for item in missing])
            if len(vectors) != len(missing):
                raise ValueError("embedding response size mismatch")
            for item, vector in zip(missing, vectors):
                self._document_embeddings[item["id"]] = np.asarray(vector, dtype=np.float32)

        reranked: list[dict[str, Any]] = []
        for candidate in candidates:
            document_vector = self._document_embeddings[candidate["id"]]
            semantic_score = _cosine_similarity(query_vector, document_vector)
            combined = 0.7 * ((semantic_score + 1.0) / 2.0) + 0.3 * float(candidate["keyword_score"])
            item = dict(candidate)
            item["semantic_score"] = float(semantic_score)
            item["score"] = float(combined)
            reranked.append(item)
        return sorted(reranked, key=lambda item: item["score"], reverse=True)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.with_suffix(".meta.json").write_text(
            json.dumps(self._documents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _terms(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    chinese_runs = re.findall(r"[\u3400-\u9fff]+", normalized)
    ascii_words = re.findall(r"[a-z0-9]+", normalized)
    terms: list[str] = list(ascii_words)
    for run in chinese_runs:
        terms.extend(run)
        terms.extend(run[index:index + 2] for index in range(len(run) - 1))
    return terms


def _term_weight(term: str) -> float:
    if re.fullmatch(r"[\u3400-\u9fff]{2}", term):
        return 2.0
    if re.fullmatch(r"[\u3400-\u9fff]", term):
        return 0.5
    return 1.5


def _idf(total_documents: int, document_frequency: int) -> float:
    return math.log((total_documents + 1) / (document_frequency + 0.5)) + 1.0


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("embedding dimensions do not match")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("embedding contains non-finite values")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        raise ValueError("zero-length embedding")
    return float(np.dot(left, right) / denominator)
