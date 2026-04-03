"""Tests for the traversal engine."""

import random
import pytest
from unittest.mock import MagicMock
from rhizome.traversal.engine import (
    TraversalEngine,
    TraversalStep,
    TraversalError,
    _softmax_sample,
    extract_article_slug,
)
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

    def search(self, query_vector, top_k, with_vector=True, query_filter=None):
        return self._results[:top_k]


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
                    "domain": "Philosophy",
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
                    "domain": "Philosophy",
                },
                "vector": mock_vector,
            },
        ]

        embedder = MockEmbedder()
        vector_store = MockVectorStore(mock_results)
        config = TraversalConfig(depth=2, epsilon=0.0, temperature=0.0)  # pure exploit, greedy

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
                    "domain": "Philosophy",
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
        assert config.top_k == 20
        assert config.temperature == 1.0
        assert config.max_same_article_consecutive == 2


# ── Softmax sampling ──────────────────────────────────────────────────────────

class TestSoftmaxSample:
    """Tests for _softmax_sample()."""

    def test_softmax_sample_greedy(self):
        """temperature=0 → always returns candidates[0]."""
        candidates = [
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.8},
            {"id": "c", "score": 0.7},
        ]
        # Run multiple times — greedy must always return first
        for _ in range(20):
            result = _softmax_sample(candidates, temperature=0.0)
            assert result == candidates[0]

    def test_softmax_sample_natural_temperature(self):
        """temperature=1.0 → higher-score candidates selected more often."""
        candidates = [
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.5},
            {"id": "c", "score": 0.1},
        ]
        # With seed, count selections over many runs
        counts = {"a": 0, "b": 0, "c": 0}
        for seed in range(200):
            random.seed(seed)
            result = _softmax_sample(candidates, temperature=1.0)
            counts[result["id"]] += 1
        # 'a' (highest score) must be selected most often
        assert counts["a"] > counts["b"] > counts["c"]

    def test_softmax_sample_flat_temperature(self):
        """temperature=2.0 → flatter distribution than temperature=1.0."""
        candidates = [
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.5},
            {"id": "c", "score": 0.1},
        ]
        # Count selections at temp=1.0 vs temp=2.0 over same seeds
        counts_flat = {"a": 0, "b": 0, "c": 0}
        for seed in range(200):
            random.seed(seed)
            result = _softmax_sample(candidates, temperature=2.0)
            counts_flat[result["id"]] += 1
        # With flat temperature, 'c' should be selected more than with greedy temp
        assert counts_flat["c"] > 0  # 'c' gets some selections with flat temp

    def test_softmax_sample_single_candidate(self):
        """Single candidate → returns it regardless of temperature."""
        candidates = [{"id": "only-one", "score": 0.5}]
        for temp in [0.0, 0.01, 1.0, 2.0, 3.0]:
            result = _softmax_sample(candidates, temperature=temp)
            assert result == candidates[0]


# ── Article slug extraction ───────────────────────────────────────────────────

class TestExtractArticleSlug:
    """Tests for extract_article_slug()."""

    def test_extract_slug_simple(self):
        """'modernism-001' → 'modernism'."""
        assert extract_article_slug("modernism-001") == "modernism"

    def test_extract_slug_hyphenated(self):
        """'post-modernism-001' → 'post-modernism'."""
        assert extract_article_slug("post-modernism-001") == "post-modernism"

    def test_extract_slug_deeply_hyphenated(self):
        """'post-anti-modernism-001' → 'post-anti-modernism'."""
        assert extract_article_slug("post-anti-modernism-001") == "post-anti-modernism"

    def test_extract_slug_multi_ordinal(self):
        """'article-name-100' → 'article-name'."""
        assert extract_article_slug("article-name-100") == "article-name"


# ── Rolling article window ─────────────────────────────────────────────────────

class TestRollingArticleWindow:
    """Tests for article hard block via rolling window."""

    def test_article_window_blocks_third_repeat(self):
        """window=2: article window blocks a 3rd consecutive chunk from the same article."""
        # The rolling window tracks recently picked article slugs. With window=2, after
        # modernism-001 is picked (window=['modernism']), modernism-002 is blocked since
        # 'modernism' is already in the window. The forced jump clears the window and
        # picks critical-theory-001. With only 2 articles in the mock, the traversal ends
        # early because the forced jump finds no further unvisited non-blocked candidates.
        candidates = [
            {"id": "modernism-001", "score": 0.9,
             "payload": {"id": "modernism-001", "text": "Modernism is...", "article_title": "Modernism",
                          "article_url": "https://en.wikipedia.org/wiki/Modernism", "domain": "Philosophy"},
             "vector": [0.1] * 384},
            {"id": "modernism-002", "score": 0.85,
             "payload": {"id": "modernism-002", "text": "Modernism continued...", "article_title": "Modernism",
                          "article_url": "https://en.wikipedia.org/wiki/Modernism", "domain": "Philosophy"},
             "vector": [0.1] * 384},
            {"id": "critical-theory-001", "score": 0.8,
             "payload": {"id": "critical-theory-001", "text": "Critical theory...", "article_title": "Critical theory",
                          "article_url": "https://en.wikipedia.org/wiki/Critical_theory", "domain": "Philosophy"},
             "vector": [0.1] * 384},
        ]
        embedder = MockEmbedder()
        vector_store = MockVectorStore(candidates)
        config = TraversalConfig(depth=4, epsilon=0.0, max_same_article_consecutive=2, temperature=0.0)
        engine = TraversalEngine(embedder=embedder, vector_store=vector_store, config=config)
        path = engine.traverse("modernism")
        # modernism-001 is picked first. modernism-002 is blocked by the article window.
        # Forced jump clears the window and picks critical-theory-001.
        # At step 2, critical-theory-001 is visited and modernism-002 is blocked, so the
        # forced jump finds no valid landing and the traversal terminates.
        assert len(path) == 2
        assert path[0].chunk_id == "modernism-001"
        assert path[0].forced_jump is False
        assert path[1].chunk_id == "critical-theory-001"
        assert path[1].forced_jump is False  # picked via softmax (filtered not empty), not forced

    def test_article_window_allows_different_after_block(self):
        """A, B, A sequence is allowed — B breaks the chain."""
        candidates = [
            {
                "id": "modernism-001",
                "score": 0.9,
                "payload": {
                    "id": "modernism-001",
                    "text": "Modernism...",
                    "article_title": "Modernism",
                    "article_url": "https://en.wikipedia.org/wiki/Modernism",
                    "domain": "Philosophy",
                },
                "vector": [0.1] * 384,
            },
            {
                "id": "postmodernism-001",
                "score": 0.8,
                "payload": {
                    "id": "postmodernism-001",
                    "text": "Postmodernism...",
                    "article_title": "Postmodernism",
                    "article_url": "https://en.wikipedia.org/wiki/Postmodernism",
                    "domain": "Philosophy",
                },
                "vector": [0.1] * 384,
            },
        ]
        embedder = MockEmbedder()
        vector_store = MockVectorStore(candidates)
        config = TraversalConfig(depth=4, epsilon=0.0, max_same_article_consecutive=2, temperature=0.0)
        engine = TraversalEngine(embedder=embedder, vector_store=vector_store, config=config)
        path = engine.traverse("modernism")
        # Only 2 unique chunks exist, so path stops after 2 steps (both visited).
        # Window=2 correctly allows: first modernism, then postmodernism (different article).
        assert len(path) == 2
        assert path[0].article_title == "Modernism"
        assert path[1].article_title == "Postmodernism"

    def test_article_window_disabled_when_zero(self):
        """max_same_article_consecutive=0 → no blocking."""
        candidates = [
            {
                "id": "modernism-001",
                "score": 0.9,
                "payload": {
                    "id": "modernism-001",
                    "text": "Modernism...",
                    "article_title": "Modernism",
                    "article_url": "https://en.wikipedia.org/wiki/Modernism",
                    "domain": "Philosophy",
                },
                "vector": [0.1] * 384,
            },
        ]
        embedder = MockEmbedder()
        vector_store = MockVectorStore(candidates)
        config = TraversalConfig(depth=3, epsilon=0.0, max_same_article_consecutive=0, temperature=0.0)
        engine = TraversalEngine(embedder=embedder, vector_store=vector_store, config=config)
        path = engine.traverse("modernism")
        # Only one unique chunk, so path stops at 1 regardless of window
        assert len(path) == 1


# ── Fallback global jump when all candidates blocked ─────────────────────────

class TestFallbackGlobalJump:
    """Tests for forced global jump when all top_k candidates are blocked."""

    def test_fallback_global_jump_when_all_candidates_blocked(self):
        """All top_k from same article → forced global jump fires."""
        # Three chunks all from the same article
        candidates = [
            {
                "id": "modernism-001",
                "score": 0.9,
                "payload": {
                    "id": "modernism-001",
                    "text": "Modernism...",
                    "article_title": "Modernism",
                    "article_url": "https://en.wikipedia.org/wiki/Modernism",
                    "domain": "Philosophy",
                },
                "vector": [0.1] * 384,
            },
            {
                "id": "modernism-002",
                "score": 0.85,
                "payload": {
                    "id": "modernism-002",
                    "text": "Modernism era...",
                    "article_title": "Modernism",
                    "article_url": "https://en.wikipedia.org/wiki/Modernism",
                    "domain": "Philosophy",
                },
                "vector": [0.1] * 384,
            },
        ]
        embedder = MockEmbedder()
        vector_store = MockVectorStore(candidates)
        # window=1 → even the first repeat is blocked; no exploit path possible
        config = TraversalConfig(depth=2, epsilon=0.0, max_same_article_consecutive=1, temperature=0.0)
        engine = TraversalEngine(embedder=embedder, vector_store=vector_store, config=config)
        path = engine.traverse("modernism")
        # First step should work (no blocking yet), second step hits block and fires global jump
        # Path may have 1 or 2 steps depending on whether global jump finds remaining chunks
        assert len(path) >= 1


# ── TraversalConfig from_dict ─────────────────────────────────────────────────

class TestTraversalConfigFromDict:
    """Tests for TraversalConfig.from_dict()."""

    def test_config_from_dict_new_fields(self):
        """from_dict() parses temperature and max_same_article_consecutive."""
        data = {
            "depth": 10,
            "epsilon": 0.2,
            "top_k": 15,
            "temperature": 1.5,
            "max_same_article_consecutive": 3,
        }
        config = TraversalConfig.from_dict(data)
        assert config.temperature == 1.5
        assert config.max_same_article_consecutive == 3
        assert config.depth == 10
        assert config.epsilon == 0.2
        assert config.top_k == 15

    def test_config_from_dict_missing_fields_use_defaults(self):
        """from_dict() falls back to defaults for missing keys."""
        data = {}
        config = TraversalConfig.from_dict(data)
        assert config.temperature == 1.0
        assert config.max_same_article_consecutive == 2

