"""Tests for CollectionManager."""

import pytest
from unittest.mock import patch, MagicMock
from qdrant_client.models import Distance
from rhizome.vectorstore.collection import CollectionManager
from rhizome.corpus.chunker import Chunk


class TestCollectionManager:
    """Tests for CollectionManager."""

    @patch("rhizome.vectorstore.collection.QdrantClient")
    def test_create_collection(self, mock_qdrant_cls):
        """create_collection calls client.create_collection with correct params."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client

        mgr = CollectionManager(url="http://localhost:6333")
        mgr.create_collection(collection_name="test-col", vector_size=1536)

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test-col"
        assert call_kwargs["vectors_config"].size == 1536
        assert call_kwargs["vectors_config"].distance == Distance.COSINE

    @patch("rhizome.vectorstore.collection.QdrantClient")
    def test_create_collection_with_recreate(self, mock_qdrant_cls):
        """recreate=True deletes collection before creating."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client

        mgr = CollectionManager(url="http://localhost:6333")
        mgr.create_collection(collection_name="test-col", vector_size=1536, recreate=True)

        mock_client.delete_collection.assert_called_once_with(collection_name="test-col")
        mock_client.create_collection.assert_called_once()

    @patch("rhizome.vectorstore.collection.QdrantClient")
    def test_upsert_chunks_maps_chunks_to_points(self, mock_qdrant_cls):
        """upsert_chunks maps Chunk objects to PointStruct correctly."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client

        mgr = CollectionManager(url="http://localhost:6333")
        chunks = [
            Chunk(
                id="modernism-001",
                text="Modernism is...",
                article_title="Modernism",
                article_url="https://en.wikipedia.org/wiki/Modernism",
                domain="Modernism",
            )
        ]
        vectors = [[0.1] * 1536]

        mgr.upsert_chunks("test-col", chunks, vectors)

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args[1]
        points = call_kwargs["points"]
        assert len(points) == 1
        # Point ID is now a UUID (derived from slug), not the slug itself
        assert points[0].id != "modernism-001"
        assert len(points[0].id) == 36  # UUID format
        assert points[0].vector == [0.1] * 1536
        # Original slug ID is preserved in payload
        assert points[0].payload["id"] == "modernism-001"
        assert points[0].payload["article_title"] == "Modernism"

    @patch("rhizome.vectorstore.collection.QdrantClient")
    def test_upsert_chunks_mismatch_raises(self, mock_qdrant_cls):
        """Mismatched chunk/vector counts raise ValueError."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client

        mgr = CollectionManager(url="http://localhost:6333")
        chunks = [Chunk(id="a", text="a", article_title="A", article_url="http://a", domain="Modernism")]
        vectors = [[0.1], [0.2]]  # 1 chunk, 2 vectors

        with pytest.raises(ValueError) as exc_info:
            mgr.upsert_chunks("test-col", chunks, vectors)
        assert "1" in str(exc_info.value) and "2" in str(exc_info.value)

    @patch("rhizome.vectorstore.collection.QdrantClient")
    def test_collection_exists_true(self, mock_qdrant_cls):
        """collection_exists returns True when collection exists."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client
        mock_client.collection_exists.return_value = True

        mgr = CollectionManager(url="http://localhost:6333")
        assert mgr.collection_exists("test-col") is True

    @patch("rhizome.vectorstore.collection.QdrantClient")
    def test_collection_exists_false(self, mock_qdrant_cls):
        """collection_exists returns False when collection does not exist."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client
        mock_client.collection_exists.return_value = False

        mgr = CollectionManager(url="http://localhost:6333")
        assert mgr.collection_exists("nonexistent-col") is False

    @patch("rhizome.vectorstore.collection.QdrantClient")
    def test_delete_chunks_by_article_uses_delete_filter(self, mock_qdrant_cls):
        """delete_chunks_by_article uses delete-by-filter (single API call)."""
        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client

        mgr = CollectionManager(url="http://localhost:6333")
        mgr.delete_chunks_by_article("test-col", "Modernism")

        # Should use delete with a Filter selector, not scroll+delete
        mock_client.delete.assert_called_once()
        call_kwargs = mock_client.delete.call_args[1]
        # points_selector should be a Filter, not a PointIdsList
        assert hasattr(call_kwargs["points_selector"], "must")
        # No scroll should have been called
        mock_client.scroll.assert_not_called()
