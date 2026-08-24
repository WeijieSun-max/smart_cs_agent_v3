from __future__ import annotations

import math
from datetime import datetime, timezone

from qdrant_client import models

from domain.customer_service_agent.interfaces.i_memory_vector_index import IMemoryVectorIndex
from domain.customer_service_agent.memory.models import MemoryType, VectorRecord, VectorSearchHit


class QdrantMemoryVectorIndex(IMemoryVectorIndex):
    def __init__(self, client, collection_name: str, *, vector_size: int, timeout_seconds: float) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout_seconds = timeout_seconds
        self._initialized = False

    @property
    def available(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
                timeout=math.ceil(self.timeout_seconds),
            )
        indexes = [
            (
                "user_id",
                models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
            ),
            ("memory_type", models.PayloadSchemaType.KEYWORD),
            ("status", models.PayloadSchemaType.KEYWORD),
            ("source_session_id", models.PayloadSchemaType.KEYWORD),
            ("updated_at", models.PayloadSchemaType.DATETIME),
            ("expires_at", models.PayloadSchemaType.DATETIME),
        ]
        for field_name, field_schema in indexes:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
                timeout=math.ceil(self.timeout_seconds),
            )
        self._initialized = True

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._require_available()
        points = [
            models.PointStruct(
                id=str(record.memory_id),
                vector=record.vector,
                payload={
                    "user_id": record.user_id,
                    "memory_type": record.memory_type.value,
                    "status": record.status.value,
                    "source_session_id": record.source_session_id,
                    "confidence": record.confidence,
                    "updated_at": _iso(record.updated_at),
                    "expires_at": _iso(record.expires_at),
                    "schema_version": record.schema_version,
                },
            )
            for record in records
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
            timeout=math.ceil(self.timeout_seconds),
        )

    def search(
        self,
        *,
        user_id: str,
        query_vector: list[float],
        memory_types: list[MemoryType],
        top_k: int,
        now: datetime,
    ) -> list[VectorSearchHit]:
        self._require_available()
        must = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(key="status", match=models.MatchValue(value="active")),
        ]
        if memory_types:
            must.append(models.FieldCondition(
                key="memory_type",
                match=models.MatchAny(any=[item.value for item in memory_types]),
            ))
        query_filter = models.Filter(
            must=must,
            should=[
                models.IsNullCondition(is_null=models.PayloadField(key="expires_at")),
                models.FieldCondition(
                    key="expires_at",
                    range=models.DatetimeRange(gt=_aware(now)),
                ),
            ],
        )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=False,
            with_vectors=False,
            timeout=math.ceil(self.timeout_seconds),
        )
        return [
            VectorSearchHit(memory_id=str(point.id), score=max(0.0, min(1.0, float(point.score))))
            for point in response.points
        ]

    def delete(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        self._require_available()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=memory_ids,
            wait=True,
            timeout=math.ceil(self.timeout_seconds),
        )

    def health_status(self) -> dict[str, object]:
        if not self._initialized:
            return {"status": "unavailable"}
        collection = self.client.get_collection(self.collection_name)
        return {"status": "ready", "collection_status": str(collection.status)}

    def _require_available(self) -> None:
        if not self._initialized:
            raise RuntimeError("Qdrant memory index is not initialized")


class DisabledMemoryVectorIndex(IMemoryVectorIndex):
    @property
    def available(self) -> bool:
        return False

    def initialize(self) -> None:
        return None

    def upsert(self, records: list[VectorRecord]) -> None:
        raise RuntimeError("Qdrant memory index is disabled")

    def search(self, **kwargs) -> list[VectorSearchHit]:
        raise RuntimeError("Qdrant memory index is disabled")

    def delete(self, memory_ids: list[str]) -> None:
        raise RuntimeError("Qdrant memory index is disabled")

    def health_status(self) -> dict[str, object]:
        return {"status": "disabled"}


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return _aware(value).isoformat().replace("+00:00", "Z") if value is not None else None
