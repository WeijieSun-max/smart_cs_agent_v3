from __future__ import annotations

import hashlib
import math
from typing import Any

from qdrant_client import models

from domain.customer_service_agent.interfaces.i_knowledge_store import IKnowledgeStore
from infra.knowledge.document_loaders import KnowledgeRecord
from infra.knowledge.sparse_encoder import LocalSparseEncoder
from pkg.exceptions.exception import ToolValidationError


class QdrantKnowledgeStore(IKnowledgeStore):
    def __init__(
        self,
        client: Any,
        collection_name: str,
        *,
        vector_size: int,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
        embedding_provider: Any | None = None,
        sparse_encoder: LocalSparseEncoder | None = None,
        embedding_mode: str = "external",
        retrieval_mode: str = "hybrid",
        sparse_encoder_name: str = "char_bigram",
        qdrant_dense_model: str = "",
        qdrant_sparse_model: str = "Qdrant/bm25",
        fusion: str = "rrf",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        prefetch_limit: int = 40,
        timeout_seconds: float = 2.0,
        dense_score_threshold: float | None = None,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.embedding_provider = embedding_provider
        self.sparse_encoder = sparse_encoder or LocalSparseEncoder()
        self.embedding_mode = embedding_mode
        self.retrieval_mode = retrieval_mode
        self.sparse_encoder_name = sparse_encoder_name
        self.qdrant_dense_model = qdrant_dense_model
        self.qdrant_sparse_model = qdrant_sparse_model
        self.fusion = fusion
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.prefetch_limit = prefetch_limit
        self.timeout_seconds = timeout_seconds
        self.dense_score_threshold = dense_score_threshold
        self._initialized = False

    def initialize(self, *, recreate: bool = False) -> None:
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(
                collection_name=self.collection_name,
                timeout=math.ceil(self.timeout_seconds),
            )
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.dense_vector_name: models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self.sparse_vector_name: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
                timeout=math.ceil(self.timeout_seconds),
            )
        for field_name in ("domain", "document_type", "source_type", "first_level_index", "status", "source", "version"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
                timeout=math.ceil(self.timeout_seconds),
            )
        self._initialized = True

    def add_document(self, content: str, source: str = "", metadata: dict | None = None) -> str:
        if not content.strip():
            raise ToolValidationError()
        record = KnowledgeRecord(
            point_id=hashlib.sha256(content.encode("utf-8")).hexdigest()[:32],
            content=content,
            payload={"source": source, "status": "active", **(metadata or {})},
        )
        self.upsert_records([record])
        return record.point_id

    def upsert_records(self, records: list[KnowledgeRecord], *, dense_vectors: list[list[float]] | None = None) -> None:
        if not records:
            return
        self._require_available()
        if self.embedding_mode == "external" and dense_vectors is None:
            dense_vectors = self._embed_documents([record.content for record in records])
        if dense_vectors is not None and len(dense_vectors) != len(records):
            raise ValueError("dense vector count does not match record count")
        points = []
        for index, record in enumerate(records):
            points.append(models.PointStruct(
                id=record.point_id,
                vector=self._record_vector(record.content, None if dense_vectors is None else dense_vectors[index]),
                payload={**record.payload, "content": record.content},
            ))
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
            timeout=math.ceil(self.timeout_seconds),
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.search_filtered(query, top_k=top_k)

    def search_filtered(self, query: str, top_k: int = 5, *, domain: str | None = None, document_type: str | None = None) -> list[dict[str, Any]]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
            raise ToolValidationError()
        if not query.strip() or len(query) > 4000:
            raise ToolValidationError()
        self._require_available()
        query_filter = self._filter(domain=domain, document_type=document_type)
        if self.retrieval_mode == "dense":
            return self._single_search(query, top_k, self.dense_vector_name, self._dense_query(query), query_filter)
        if self.retrieval_mode == "sparse":
            return self._single_search(query, top_k, self.sparse_vector_name, self._sparse_query(query), query_filter)
        if self.fusion == "client_weighted_rrf":
            return self._client_weighted_rrf(query, top_k, query_filter)
        return self._qdrant_fusion_search(query, top_k, query_filter)

    def health_status(self) -> dict[str, object]:
        if not self._initialized:
            return {"status": "unavailable", "mode": "qdrant"}
        collection = self.client.get_collection(self.collection_name)
        return {
            "status": "ready",
            "mode": f"qdrant-{self.retrieval_mode}",
            "embedding_mode": self.embedding_mode,
            "collection_status": str(collection.status),
        }

    def _qdrant_fusion_search(self, query: str, top_k: int, query_filter: models.Filter) -> list[dict[str, Any]]:
        prefetch = [
            self._prefetch(self.sparse_vector_name, self._sparse_query(query), query_filter),
            self._prefetch(self.dense_vector_name, self._dense_query(query), query_filter, score_threshold=self.dense_score_threshold),
        ]
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=self._fusion_query(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            timeout=math.ceil(self.timeout_seconds),
        )
        return self._points_to_results(response.points)

    def _single_search(self, top_k: int, using: str, query: Any, query_filter: models.Filter) -> list[dict[str, Any]]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            using=using,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            timeout=math.ceil(self.timeout_seconds),
        )
        return self._points_to_results(response.points)

    def _client_weighted_rrf(self, query: str, top_k: int, query_filter: models.Filter) -> list[dict[str, Any]]:
        dense_hits = self._single_search(self.prefetch_limit, self.dense_vector_name, self._dense_query(query), query_filter)
        sparse_hits = self._single_search(self.prefetch_limit, self.sparse_vector_name, self._sparse_query(query), query_filter)
        by_id: dict[str, dict[str, Any]] = {}
        rrf_k = 60.0
        for rank, hit in enumerate(dense_hits, start=1):
            item = by_id.setdefault(str(hit["id"]), dict(hit, score=0.0))
            item["score"] += self.dense_weight / (rrf_k + rank)
            item["dense_score"] = hit["score"]
        for rank, hit in enumerate(sparse_hits, start=1):
            item = by_id.setdefault(str(hit["id"]), dict(hit, score=0.0))
            item["score"] += self.sparse_weight / (rrf_k + rank)
            item["sparse_score"] = hit["score"]
        return sorted(by_id.values(), key=lambda item: item["score"], reverse=True)[:top_k]

    def _record_vector(self, content: str, dense_vector: list[float] | None) -> dict[str, Any]:
        if self.embedding_mode == "qdrant":
            return {
                self.dense_vector_name: self._document_query(content, self._dense_model()),
                self.sparse_vector_name: self._document_query(content, self.qdrant_sparse_model),
            }
        if dense_vector is None:
            raise ValueError("dense vector is required for external embedding mode")
        return {
            self.dense_vector_name: dense_vector,
            self.sparse_vector_name: self.sparse_encoder.encode(content),
        }

    def _dense_query(self, query: str) -> Any:
        if self.embedding_mode == "qdrant":
            return self._document_query(query, self._dense_model())
        if self.embedding_provider is None:
            raise RuntimeError("embedding provider is required for external knowledge search")
        return self.embedding_provider.embed_query(query)

    def _sparse_query(self, query: str) -> Any:
        if self.embedding_mode == "qdrant" and self.sparse_encoder_name == "qdrant_bm25":
            return self._document_query(query, self.qdrant_sparse_model)
        return self.sparse_encoder.encode(query)

    def _embed_documents(self, contents: list[str]) -> list[list[float]]:
        if self.embedding_provider is None:
            raise RuntimeError("embedding provider is required for external knowledge indexing")
        return self.embedding_provider.embed_documents(contents)

    def _prefetch(self, using: str, query: Any, query_filter: models.Filter, *, score_threshold: float | None = None) -> models.Prefetch:
        fields = getattr(models.Prefetch, "model_fields", {})
        kwargs: dict[str, Any] = {"query": query, "using": using, "limit": self.prefetch_limit}
        if "filter" in fields:
            kwargs["filter"] = query_filter
        elif "query_filter" in fields:
            kwargs["query_filter"] = query_filter
        if score_threshold is not None and "score_threshold" in fields:
            kwargs["score_threshold"] = score_threshold
        return models.Prefetch(**kwargs)

    def _fusion_query(self) -> Any:
        if self.fusion == "dbsf":
            return models.FusionQuery(fusion=models.Fusion.DBSF)
        if self.fusion == "weighted_rrf":
            return {"rrf": {"weights": [self.sparse_weight, self.dense_weight]}}
        return models.FusionQuery(fusion=models.Fusion.RRF)

    def _filter(self, *, domain: str | None, document_type: str | None) -> models.Filter:
        must: list[Any] = [models.FieldCondition(key="status", match=models.MatchValue(value="active"))]
        if domain is not None:
            must.append(models.FieldCondition(key="domain", match=models.MatchValue(value=domain)))
        if document_type is not None:
            must.append(models.FieldCondition(key="document_type", match=models.MatchValue(value=document_type)))
        return models.Filter(must=must)

    def _document_query(self, text: str, model: str) -> models.Document:
        if not model.strip():
            raise RuntimeError("Qdrant inference model is required")
        return models.Document(text=text, model=model)

    def _dense_model(self) -> str:
        return self.qdrant_dense_model

    def _points_to_results(self, points: list[Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            content = str(payload.pop("content", ""))
            results.append({
                "id": str(point.id),
                "content": content,
                "source": payload.get("source", ""),
                "metadata": payload,
                "score": float(point.score),
            })
        return results

    def _require_available(self) -> None:
        if not self._initialized:
            raise RuntimeError("Qdrant knowledge store is not initialized")
