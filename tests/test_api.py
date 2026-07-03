"""API contract tests for the FastAPI surface.

Uses FastAPI's TestClient (sync) with `app.dependency_overrides` to swap in
fake embedder, vector store, and config. No real network calls.

Contract coverage:
  /health      — Qdrant reachable / unreachable
  /config      — returns wikipedia_categories
  /traverse    — happy path, missing collection, traversal error, embedding
                 error, validation (422) for out-of-range params
  /traverse/stream — emitted SSE events are well-formed

Each test asserts shape and status, not implementation details.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rhizome.api.main import (
    app,
    get_config_dep,
    get_embedder_dep,
    get_vector_store_dep,
)
from rhizome.config import RhizomeConfig
from rhizome.embedder.base import EmbeddingError
from rhizome.traversal.engine import TraversalError, TraversalStep


# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────


class FakeEmbedder:
    """Drop-in replacement for Embedder. Returns canned vectors.

    Vector size is configurable so tests can match production (1536) or use
    a smaller value to keep the response payload inspectable.
    """

    def __init__(self, vector_size: int = 4) -> None:
        self.vector_size = vector_size
        self.embed_calls: list[list[str]] = []
        # Stable canned vector so similarity is predictable
        self._vector = [0.1] * vector_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [list(self._vector) for _ in texts]


class FakeVectorStore:
    """Drop-in replacement for VectorStoreClient.

    Mimics the subset of the Qdrant client used by the API: `collection_exists`
    and `get_collection`. The traversal engine also calls this through
    `search` / `search_excluding`, but the engine is mocked at a higher
    boundary (FakeTraversalEngine) for unit-level API tests.
    """

    def __init__(
        self,
        collection_exists: bool = True,
        get_collection_raises: bool = False,
    ) -> None:
        self.collection_exists_return = collection_exists
        self.get_collection_raises = get_collection_raises
        self._client = _FakeQdrantClient(
            collection_exists=collection_exists,
            get_collection_raises=get_collection_raises,
        )

    @property
    def client(self) -> "_FakeQdrantClient":
        return self._client


class _FakeQdrantClient:
    """Mocks the qdrant_client.QdrantClient surface used by the API."""

    def __init__(self, collection_exists: bool, get_collection_raises: bool) -> None:
        self._collection_exists = collection_exists
        self._get_collection_raises = get_collection_raises
        self.collection_exists_calls: list[str] = []

    def collection_exists(self, collection_name: str) -> bool:
        self.collection_exists_calls.append(collection_name)
        if self._get_collection_raises:
            raise ConnectionError("Qdrant unreachable")
        return self._collection_exists

    def get_collection(self, collection_name: str) -> dict:
        if self._get_collection_raises:
            raise ConnectionError("Qdrant unreachable")
        return {"name": collection_name}


class FakeTraversalEngine:
    """Fake engine with a configurable traverse() and traverse_stream()."""

    def __init__(
        self,
        path: list[TraversalStep] | None = None,
        traverse_raises: Exception | None = None,
    ) -> None:
        self._path = path or []
        self._traverse_raises = traverse_raises
        self.path: list[str] = []
        self.traverse_calls: list[str] = []

    def traverse(self, query: str) -> list[TraversalStep]:
        self.traverse_calls.append(query)
        if self._traverse_raises is not None:
            raise self._traverse_raises
        return list(self._path)

    async def traverse_stream(self, query: str):
        self.traverse_calls.append(query)
        for step in self._path:
            self.path.append(step.chunk_id)
            yield step


def make_traversal_step(
    chunk_id: str = "modernism-001",
    text: str = "Modernism is a philosophical movement.",
    article_title: str = "Modernism",
    article_url: str = "https://en.wikipedia.org/wiki/Modernism",
    similarity: float = 0.95,
    forced_jump: bool = False,
    with_candidate: bool = True,
) -> TraversalStep:
    """Build a TraversalStep with sensible defaults."""
    candidates: list[dict] = []
    if with_candidate:
        candidates = [
            {
                "id": chunk_id,
                "score": similarity,
                "payload": {
                    "id": chunk_id,
                    "text": text,
                    "article_title": article_title,
                    "article_url": article_url,
                },
            }
        ]
    return TraversalStep(
        chunk_id=chunk_id,
        text=text,
        article_title=article_title,
        article_url=article_url,
        depth=0,
        similarity=similarity,
        forced_jump=forced_jump,
        candidates=candidates,
    )


def make_config(**overrides: Any) -> RhizomeConfig:
    """Build a RhizomeConfig with required fields filled in.

    Any field can be overridden via kwargs. Required: qdrant_collection.
    """
    defaults: dict[str, Any] = {
        "qdrant_url": "http://localhost:6333",
        "qdrant_collection": "modernity-v1",
        "qdrant_api_key": None,
        "embedder_type": "openai",
        "openai_api_key": "sk-test",
        "hf_api_token": None,
        "hf_model": "sentence-transformers/all-MiniLM-L6-v2",
        "wikipedia_categories": "Modernism,Postmodernism",
        "default_depth": 8,
        "epsilon": 0.1,
    }
    defaults.update(overrides)
    return RhizomeConfig(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: TestClient with overridable dependencies
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    """Bare TestClient. Tests compose their own overrides via app.dependency_overrides."""
    return TestClient(app)


@pytest.fixture
def config() -> RhizomeConfig:
    """Default config used by most tests."""
    return make_config()


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────


class TestHealth:
    def test_returns_200_when_qdrant_reachable(self, client, config):
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.get("/health")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200

    def test_returns_503_when_qdrant_unreachable(self, client, config):
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            get_collection_raises=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.get("/health")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 503
        assert response.json()["detail"] == "Qdrant unavailable"


# ─────────────────────────────────────────────────────────────────────────────
# /config
# ─────────────────────────────────────────────────────────────────────────────


class TestConfig:
    def test_returns_categories(self, client):
        app.dependency_overrides[get_config_dep] = lambda: make_config(
            wikipedia_categories="Modernism,Postmodernism,Critical theory"
        )
        try:
            response = client.get("/config")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json() == {
            "categories": "Modernism,Postmodernism,Critical theory"
        }

    def test_empty_categories_returns_empty_string(self, client):
        app.dependency_overrides[get_config_dep] = lambda: make_config(
            wikipedia_categories=""
        )
        try:
            response = client.get("/config")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json() == {"categories": ""}


# ─────────────────────────────────────────────────────────────────────────────
# /traverse
# ─────────────────────────────────────────────────────────────────────────────


class TestTraverseHappyPath:
    def test_returns_path_and_stats(self, client, config, monkeypatch):
        step = make_traversal_step()
        fake_engine = FakeTraversalEngine(path=[step])
        monkeypatch.setattr(
            "rhizome.api.main.TraversalEngine",
            lambda **kwargs: fake_engine,
        )
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.post(
                "/traverse",
                json={"query": "modernism", "depth": 5},
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        body = response.json()
        assert "path" in body and "stats" in body
        assert len(body["path"]) == 1
        assert body["path"][0]["chunk_id"] == "modernism-001"
        assert body["path"][0]["text"] == "Modernism is a philosophical movement."
        assert body["path"][0]["forced_jump"] is False
        assert body["path"][0]["similarity"] == pytest.approx(0.95)
        # candidates propagated
        assert len(body["path"][0]["candidates"]) == 1
        assert body["path"][0]["candidates"][0]["chunk_id"] == "modernism-001"
        # stats
        assert body["stats"]["depth"] == 5
        assert body["stats"]["forced_jumps"] == 0
        assert body["stats"]["categories"] == "Modernism,Postmodernism"

    def test_uses_default_params_when_omitted(self, client, config, monkeypatch):
        fake_engine = FakeTraversalEngine(path=[])
        monkeypatch.setattr(
            "rhizome.api.main.TraversalEngine",
            lambda **kwargs: fake_engine,
        )
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.post("/traverse", json={"query": "x"})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200
        body = response.json()
        # Defaults from TraverseRequest
        assert body["stats"]["depth"] == 8
        assert body["stats"]["epsilon"] == pytest.approx(0.1)
        assert body["stats"]["top_k"] == 20

    def test_counts_forced_jumps_in_stats(self, client, config, monkeypatch):
        normal = make_traversal_step(chunk_id="a", forced_jump=False)
        jumped = make_traversal_step(chunk_id="b", forced_jump=True)
        fake_engine = FakeTraversalEngine(path=[normal, normal, jumped])
        monkeypatch.setattr(
            "rhizome.api.main.TraversalEngine",
            lambda **kwargs: fake_engine,
        )
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.post("/traverse", json={"query": "x"})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200
        assert response.json()["stats"]["forced_jumps"] == 1


class TestTraverseErrorPaths:
    def test_returns_400_when_collection_missing(self, client, config, monkeypatch):
        fake_engine = FakeTraversalEngine(path=[])
        monkeypatch.setattr(
            "rhizome.api.main.TraversalEngine",
            lambda **kwargs: fake_engine,
        )
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=False
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.post("/traverse", json={"query": "x"})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]

    def test_returns_503_when_qdrant_unreachable(self, client, config):
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            get_collection_raises=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.post("/traverse", json={"query": "x"})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 503
        assert response.json()["detail"] == "Qdrant unavailable"

    def test_returns_500_on_traversal_error(self, client, config, monkeypatch):
        fake_engine = FakeTraversalEngine(traverse_raises=TraversalError("empty"))
        monkeypatch.setattr(
            "rhizome.api.main.TraversalEngine",
            lambda **kwargs: fake_engine,
        )
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.post("/traverse", json={"query": "x"})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 500
        assert "Traversal failed" in response.json()["detail"]

    def test_returns_500_on_embedding_error(self, client, config, monkeypatch):
        fake_engine = FakeTraversalEngine(
            traverse_raises=EmbeddingError("OpenAI API error: 401")
        )
        monkeypatch.setattr(
            "rhizome.api.main.TraversalEngine",
            lambda **kwargs: fake_engine,
        )
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.post("/traverse", json={"query": "x"})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 500
        assert "Embedding error" in response.json()["detail"]


class TestTraverseValidation:
    """Pydantic-validated fields surface as 422 from FastAPI."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("depth", 0),       # ge=1
            ("depth", 101),     # le=100
            ("epsilon", -0.1),  # ge=0
            ("epsilon", 1.5),   # le=1
            ("top_k", 0),       # ge=1
            ("top_k", 51),      # le=50
            ("temperature", -0.1),  # ge=0
            ("temperature", 3.1),   # le=3
            ("max_same_article_consecutive", -1),  # ge=0
            ("max_same_article_consecutive", 21),  # le=20
        ],
    )
    def test_out_of_range_field_returns_422(self, client, config, monkeypatch, field, value):
        # Need a working Qdrant to get past the 400 check; validation runs first
        # but we still wire up overrides so the test is realistic.
        fake_engine = FakeTraversalEngine(path=[])
        monkeypatch.setattr(
            "rhizome.api.main.TraversalEngine",
            lambda **kwargs: fake_engine,
        )
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.post(
                "/traverse",
                json={"query": "x", field: value},
            )
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 422

    def test_empty_query_returns_422(self, client, config):
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.post("/traverse", json={"query": ""})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 422

    def test_query_over_500_chars_returns_422(self, client, config):
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.post("/traverse", json={"query": "a" * 501})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 422

    def test_missing_query_returns_422(self, client, config):
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.post("/traverse", json={})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# /traverse/stream
# ─────────────────────────────────────────────────────────────────────────────


class TestTraverseStream:
    def _read_sse_events(self, response) -> list[dict]:
        """Parse an SSE response body into a list of decoded JSON events."""
        events: list[dict] = []
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if not payload.strip():
                continue
            events.append(json.loads(payload))
        return events

    def test_streams_step_and_done_events(self, client, config, monkeypatch):
        steps = [
            make_traversal_step(chunk_id=f"step-{i}", text=f"text {i}")
            for i in range(3)
        ]
        fake_engine = FakeTraversalEngine(path=steps)
        monkeypatch.setattr(
            "rhizome.api.main.TraversalEngine",
            lambda **kwargs: fake_engine,
        )
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            with client.stream("POST", "/traverse/stream", json={"query": "x"}) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                events = self._read_sse_events(response)
        finally:
            app.dependency_overrides.clear()

        # One step event per TraversalStep + one done event
        step_events = [e for e in events if e["type"] == "step"]
        done_events = [e for e in events if e["type"] == "done"]
        assert len(step_events) == 3
        assert len(done_events) == 1

        # First step event carries correct shape
        first = step_events[0]
        assert first["type"] == "step"
        assert first["chunk_id"] == "step-0"
        assert first["text"] == "text 0"
        assert first["depth"] == 0
        assert first["similarity"] == pytest.approx(0.95)
        assert first["forced_jump"] is False
        assert isinstance(first["candidates"], list)

        # Done event carries accumulated path + stats
        done = done_events[0]
        assert done["type"] == "done"
        assert done["path"] == ["step-0", "step-1", "step-2"]
        assert done["stats"]["depth"] == 8
        assert done["stats"]["forced_jumps"] == 0

    def test_done_event_reports_forced_jumps(self, client, config, monkeypatch):
        steps = [
            make_traversal_step(chunk_id="a", forced_jump=False),
            make_traversal_step(chunk_id="b", forced_jump=True),
            make_traversal_step(chunk_id="c", forced_jump=True),
        ]
        fake_engine = FakeTraversalEngine(path=steps)
        monkeypatch.setattr(
            "rhizome.api.main.TraversalEngine",
            lambda **kwargs: fake_engine,
        )
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            collection_exists=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            with client.stream("POST", "/traverse/stream", json={"query": "x"}) as response:
                events = self._read_sse_events(response)
        finally:
            app.dependency_overrides.clear()

        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["stats"]["forced_jumps"] == 2

    def test_503_when_qdrant_unreachable(self, client, config):
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_vector_store_dep] = lambda: FakeVectorStore(
            get_collection_raises=True
        )
        app.dependency_overrides[get_config_dep] = lambda: config
        try:
            response = client.post("/traverse/stream", json={"query": "x"})
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 503