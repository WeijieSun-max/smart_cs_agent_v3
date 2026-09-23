from types import SimpleNamespace

from infra.knowledge.document_loaders import KnowledgeRecord
from infra.knowledge.qdrant_knowledge_store import QdrantKnowledgeStore


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.created_collection = None
        self.payload_indexes: list[str] = []
        self.upserted_points = []
        self.queries = []
        self.exists = False

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs):
        self.created_collection = kwargs
        self.exists = True

    def delete_collection(self, **kwargs):
        self.exists = False

    def create_payload_index(self, **kwargs):
        self.payload_indexes.append(kwargs["field_name"])

    def upsert(self, **kwargs):
        self.upserted_points.extend(kwargs["points"])

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        payload = {
            "content": "一级分类：账户使用类\n问题：话费查询介绍\n答案：拨打10086。",
            "source": "data/10086_qa/01_账户使用类.md",
            "domain": "telecom",
            "document_type": "troubleshooting",
            "status": "active",
        }
        return SimpleNamespace(points=[SimpleNamespace(id="p1", score=0.9, payload=payload)])

    def get_collection(self, collection_name: str):
        return SimpleNamespace(status="green")


def _store(client: FakeQdrantClient, **kwargs) -> QdrantKnowledgeStore:
    store = QdrantKnowledgeStore(
        client,
        "knowledge_10086_v1",
        vector_size=3,
        embedding_provider=FakeEmbedder(),
        **kwargs,
    )
    store.initialize()
    return store


def test_qdrant_knowledge_store_initializes_named_vectors_and_payload_indexes() -> None:
    client = FakeQdrantClient()
    _store(client)

    assert "dense" in client.created_collection["vectors_config"]
    assert "sparse" in client.created_collection["sparse_vectors_config"]
    assert {"domain", "document_type", "source_type", "first_level_index", "status", "source", "version"}.issubset(client.payload_indexes)


def test_qdrant_knowledge_store_upserts_payload_and_named_vectors() -> None:
    client = FakeQdrantClient()
    store = _store(client)
    record = KnowledgeRecord(
        point_id="point-1",
        content="一级分类：账户使用类\n问题：话费查询\n答案：拨打10086。",
        payload={"domain": "telecom", "document_type": "troubleshooting", "status": "active"},
    )

    store.upsert_records([record])

    point = client.upserted_points[0]
    assert point.id == "point-1"
    assert "dense" in point.vector
    assert "sparse" in point.vector
    assert point.payload["content"] == record.content
    assert point.payload["domain"] == "telecom"


def test_qdrant_knowledge_store_hybrid_search_uses_prefetch_and_payload_filter() -> None:
    client = FakeQdrantClient()
    store = _store(client)

    results = store.search_filtered("话费查询", domain="telecom", document_type="troubleshooting")

    query = client.queries[0]
    assert len(query["prefetch"]) == 2
    assert query["limit"] == 5
    assert results[0]["metadata"]["domain"] == "telecom"
    assert results[0]["content"].startswith("一级分类")


def test_qdrant_knowledge_store_client_weighted_rrf_merges_dense_and_sparse() -> None:
    client = FakeQdrantClient()
    store = _store(client, fusion="client_weighted_rrf")

    results = store.search("话费查询", top_k=3)

    assert len(client.queries) == 2
    assert results[0]["id"] == "p1"
    assert results[0]["score"] > 0
