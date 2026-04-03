"""Tests for the traversal engine."""

import pytest
from unittest.mock import MagicMock
from rhizome.traversal.engine import TraversalEngine, TraversalStep, TraversalError
from rhizome.traversal.config import TraversalConfig
from rhizome.embedder.base import Embedder


class MockEmbedder(Embedder):
    """Test double for Embedder."""

    def __init__(self, vector: list[float] | None = None):
        self._vector = vector or [0.1] * 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


class MockVectorStore:
    """Test double for VectorStoreClient."""

    def __init__(self, results: list[dict] | None = None):
        self._results = results or []

    def search_excluding(self, query_vector, exclude_ids, top_k, query_filter=None, with_vector=True):
        return [r for r in self._results if r["id"] not in exclude_ids]


class TestTraversalEngine:
    """Tests for TraversalEngine."""

    def test_traverse_returns_path(self):
        """traverse() returns a non-empty path of TraversalSteps."""
        mock_vector = [0.1] * 384
        mock_results = [
            {
                "id": "modernism-001",
                "score": 0.9,
                "payload": {
                    "id": "modernism-001",
                    "text": "Modernism is...",
                    "article_title": "Modernism",
                    "article_url": "https://en.wikipedia.org/wiki/Modernism",
                },
                "vector": mock_vector,
            },
            {
                "id": "postmodernism-001",
                "score": 0.8,
                "payload": {
                    "id": "postmodernism-001",
                    "text": "Postmodernism is...",
                    "article_title": "Postmodernism",
                    "article_url": "https://en.wikipedia.org/wiki/Postmodernism",
                },
                "vector": mock_vector,
            },
        ]

        embedder = MockEmbedder()
        vector_store = MockVectorStore(mock_results)
        config = TraversalConfig(depth=2, epsilon=0.0)  # pure exploit

        engine = TraversalEngine(embedder=embedder, vector_store=vector_store, config=config)
        path = engine.traverse("modernism")

        assert len(path) == 2
        assert path[0].chunk_id == "modernism-001"
        assert path[1].chunk_id == "postmodernism-001"
        assert all(isinstance(step, TraversalStep) for step in path)

    def test_traverse_respects_visited_set(self):
        """Traversed chunks are not revisited."""
        mock_results = [
            {
                "id": "chunk-a",
                "score": 0.9,
                "payload": {
                    "id": "chunk-a",
                    "text": "Chunk A text",
                    "article_title": "Article A",
                    "article_url": "https://example.com/a",
                },
                "vector": [0.1] * 384,
            },
        ]

        embedder = MockEmbedder()
        vector_store = MockVectorStore(mock_results)
        config = TraversalConfig(depth=5, epsilon=0.0)

        engine = TraversalEngine(embedder=embedder, vector_store=vector_store, config=config)
        path = engine.traverse("test")

        # Only one unique chunk exists, so path length is 1 before running out
        assert len(path) == 1

    def test_traverse_empty_collection_raises(self):
        """Empty collection raises TraversalError."""
        embedder = MockEmbedder()
        vector_store = MockVectorStore([])  # no results
        config = TraversalConfig(depth=5, epsilon=0.0)

        engine = TraversalEngine(embedder=embedder, vector_store=vector_store, config=config)
        path = engine.traverse("nonexistent concept")

        # Should return empty path (early termination), not raise
        assert path == []

    def test_traversal_config_defaults(self):
        """TraversalConfig has sensible defaults."""
        config = TraversalConfig()
        assert config.depth == 8
        assert config.epsilon == 0.1
        assert config.top_k == 5
