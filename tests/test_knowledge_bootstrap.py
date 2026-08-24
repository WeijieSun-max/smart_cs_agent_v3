from pathlib import Path
from types import SimpleNamespace

from infra.knowledge import bootstrap


def _settings():
    return SimpleNamespace(
        knowledge_qdrant_enabled=True,
        knowledge_qdrant_collection="knowledge-test",
        root_dir=Path("D:/tmp/project"),
        vector_store_path="data/vector_store/test",
        embedding_dim=8,
        rag_keyword_min_score=0.1,
        rag_candidate_limit=4,
    )


def test_qdrant_initialization_failure_falls_back_to_local_store(monkeypatch) -> None:
    created = []
    initialized = []

    class LocalStore:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

    monkeypatch.setattr(bootstrap, "get_settings", _settings)
    monkeypatch.setattr(bootstrap, "_embedding_provider", lambda _settings: None)
    monkeypatch.setattr(
        bootstrap,
        "_initialize_qdrant_store",
        lambda *_args: (_ for _ in ()).throw(ConnectionError("qdrant unavailable")),
    )
    monkeypatch.setattr(bootstrap, "LocalKnowledgeStore", LocalStore)
    monkeypatch.setattr(bootstrap, "_seed_knowledge", lambda store: None)
    monkeypatch.setattr(bootstrap.knowledge_service, "initialize_service", initialized.append)

    bootstrap.initialize_knowledge_store()

    assert len(created) == 1
    assert initialized == created
    assert created[0].kwargs["index_path"] == Path("D:/tmp/project/data/vector_store/test")


def test_available_qdrant_store_remains_preferred(monkeypatch) -> None:
    qdrant_store = object()
    initialized = []

    monkeypatch.setattr(bootstrap, "get_settings", _settings)
    monkeypatch.setattr(bootstrap, "_embedding_provider", lambda _settings: None)
    monkeypatch.setattr(bootstrap, "_initialize_qdrant_store", lambda *_args: qdrant_store)
    monkeypatch.setattr(
        bootstrap,
        "LocalKnowledgeStore",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("local fallback must not initialize")),
    )
    monkeypatch.setattr(bootstrap.knowledge_service, "initialize_service", initialized.append)

    bootstrap.initialize_knowledge_store()

    assert initialized == [qdrant_store]
