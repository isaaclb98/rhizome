"""Tests for VectorStoreClient."""

import pytest
from unittest.mock import patch, MagicMock
from rhizome.vectorstore.client import VectorStoreClient


class TestVectorStoreClient:
    """Tests for VectorStoreClient."""

    @patch("rhizome.vectorstore.client.QdrantClient")
    def test_search_returns_results(self, mock_qdrant_cls):
        """search() returns list of dicts with id, score, payload."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client

        mock_hit = MagicMock()
        mock_hit.id = "modernism-001"
        mock_hit.score = 0.95
        mock_hit.payload = {"id": "modernism-001", "text": "Modernism is..."}
        mock_client.search.return_value = [mock_hit]

        client = VectorStoreClient(url="http://localhost:6333", collection_name="test-col")
        results = client.search(query_vector=[0.1] * 1536, top_k=5)

        assert len(results) == 1
        assert results[0]["id"] == "modernism-001"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"]["text"] == "Modernism is..."

    @patch("rhizome.vectorstore.client.QdrantClient")
    def test_search_empty_results(self, mock_qdrant_cls):
        """search() returns empty list when no matches."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client
        mock_client.search.return_value = []

        client = VectorStoreClient(url="http://localhost:6333", collection_name="test-col")
        results = client.search(query_vector=[0.1] * 1536, top_k=5)

        assert results == []

    @patch("rhizome.vectorstore.client.QdrantClient")
    def test_search_excluding_removes_ids(self, mock_qdrant_cls):
        """search_excluding() removes excluded chunk IDs from results."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client

        hit1 = MagicMock()
        hit1.id = "modernism-001"
        hit1.score = 0.95
        hit1.payload = {"id": "modernism-001", "text": "Modernism is..."}

        hit2 = MagicMock()
        hit2.id = "modernism-002"
        hit2.score = 0.90
        hit2.payload = {"id": "modernism-002", "text": "Modernism section 2..."}

        hit3 = MagicMock()
        hit3.id = "postmodernism-001"
        hit3.score = 0.85
        hit3.payload = {"id": "postmodernism-001", "text": "Postmodernism is..."}

        mock_client.search.return_value = [hit1, hit2, hit3]

        client = VectorStoreClient(url="http://localhost:6333", collection_name="test-col")
        results = client.search_excluding(
            query_vector=[0.1] * 1536,
            exclude_ids=["modernism-001"],
            top_k=5,
        )

        ids = [r["id"] for r in results]
        assert "modernism-001" not in ids
        assert "modernism-002" in ids
        assert "postmodernism-001" in ids

    @patch("rhizome.vectorstore.client.QdrantClient")
    def test_search_excluding_all_excluded(self, mock_qdrant_cls):
        """search_excluding() returns fewer results when all top results are excluded."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client

        hit1 = MagicMock()
        hit1.id = "chunk-a"
        hit1.score = 0.95
        hit1.payload = {"id": "chunk-a", "text": "Text A"}

        mock_client.search.return_value = [hit1]

        client = VectorStoreClient(url="http://localhost:6333", collection_name="test-col")
        results = client.search_excluding(
            query_vector=[0.1] * 1536,
            exclude_ids=["chunk-a"],  # Exclude the only result
            top_k=5,
        )

        assert results == []

    @patch("rhizome.vectorstore.client.QdrantClient")
    def test_search_excluding_uses_top_k_times_3(self, mock_qdrant_cls):
        """search_excluding() fetches top_k * 3 to account for exclusions."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client
        mock_client.search.return_value = []

        client = VectorStoreClient(url="http://localhost:6333", collection_name="test-col")
        client.search_excluding(query_vector=[0.1] * 1536, exclude_ids=[], top_k=5)

        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["limit"] == 15  # top_k * 3
