"""Tests for the embedder module."""

import pytest
from unittest.mock import patch, MagicMock
from rhizome.embedder.huggingface import HuggingFaceEmbedder, EmbeddingError


class TestHuggingFaceEmbedder:
    """Tests for HuggingFaceEmbedder."""

    @patch("rhizome.embedder.huggingface.requests.post")
    def test_embed_returns_vectors(self, mock_post):
        """embed() returns a list of embedding vectors."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [[0.1, 0.2, 0.3] * 128]  # 384 dims
        mock_post.return_value = mock_response

        embedder = HuggingFaceEmbedder(api_token="test-token")
        result = embedder.embed(["test text"])

        assert len(result) == 1
        assert len(result[0]) == 384
        mock_post.assert_called_once()

    @patch("rhizome.embedder.huggingface.requests.post")
    def test_embed_batch(self, mock_post):
        """Batch embedding returns correct number of vectors."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [0.1] * 384,
            [0.2] * 384,
        ]
        mock_post.return_value = mock_response

        embedder = HuggingFaceEmbedder(api_token="test-token")
        result = embedder.embed(["text one", "text two"])

        assert len(result) == 2
        assert len(result[0]) == 384
        assert len(result[1]) == 384

    @patch("rhizome.embedder.huggingface.requests.post")
    def test_embed_api_error(self, mock_post):
        """API error raises EmbeddingError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited"
        mock_post.return_value = mock_response

        embedder = HuggingFaceEmbedder(api_token="test-token")
        with pytest.raises(EmbeddingError) as exc_info:
            embedder.embed(["test"])
        assert "429" in str(exc_info.value)

    def test_vector_size(self):
        """vector_size() returns correct dimension for all-MiniLM-L6-v2."""
        embedder = HuggingFaceEmbedder()
        assert embedder.vector_size() == 384
