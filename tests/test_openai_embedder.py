"""Tests for OpenAIEmbedder."""

import pytest
from unittest.mock import patch, MagicMock
from rhizome.embedder.openai import OpenAIEmbedder, EmbeddingError


class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder."""

    @patch("rhizome.embedder.openai.requests.post")
    def test_embed_returns_vectors(self, mock_post):
        """embed() returns a list of embedding vectors."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1] * 1536}
            ]
        }
        mock_post.return_value = mock_response

        embedder = OpenAIEmbedder(api_key="test-key")
        result = embedder.embed(["test text"])

        assert len(result) == 1
        assert len(result[0]) == 1536

    @patch("rhizome.embedder.openai.requests.post")
    def test_embed_batch(self, mock_post):
        """Batch embedding returns correct number of vectors."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1] * 1536},
                {"embedding": [0.2] * 1536},
            ]
        }
        mock_post.return_value = mock_response

        embedder = OpenAIEmbedder(api_key="test-key")
        result = embedder.embed(["text one", "text two"])

        assert len(result) == 2
        assert len(result[0]) == 1536
        assert len(result[1]) == 1536

    @patch("rhizome.embedder.openai.requests.post")
    def test_embed_api_error(self, mock_post):
        """API error raises EmbeddingError with status code."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_post.return_value = mock_response

        embedder = OpenAIEmbedder(api_key="test-key")
        with pytest.raises(EmbeddingError) as exc_info:
            embedder.embed(["test"])
        assert "429" in str(exc_info.value)

    @patch("rhizome.embedder.openai.requests.post")
    def test_embed_unauthorized_error(self, mock_post):
        """401 raises EmbeddingError."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        mock_post.return_value = mock_response

        embedder = OpenAIEmbedder(api_key="bad-key")
        with pytest.raises(EmbeddingError) as exc_info:
            embedder.embed(["test"])
        assert "401" in str(exc_info.value)

    def test_vector_size_text_embedding_3_small(self):
        """vector_size() returns 1536 for text-embedding-3-small."""
        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        assert embedder.vector_size() == 1536

    def test_vector_size_text_embedding_3_large(self):
        """vector_size() returns 3072 for text-embedding-3-large."""
        embedder = OpenAIEmbedder(model="text-embedding-3-large")
        assert embedder.vector_size() == 3072

    def test_vector_size_ada_002(self):
        """vector_size() returns 1536 for text-embedding-ada-002."""
        embedder = OpenAIEmbedder(model="text-embedding-ada-002")
        assert embedder.vector_size() == 1536

    def test_vector_size_unknown_model_defaults_to_1536(self):
        """Unknown model defaults to 1536."""
        embedder = OpenAIEmbedder(model="unknown-model")
        assert embedder.vector_size() == 1536
