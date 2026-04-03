"""Tests for the embedder factory."""

import pytest
from rhizome.embedder.factory import get_embedder
from rhizome.embedder import OpenAIEmbedder, HuggingFaceEmbedder, EmbeddingError


class TestEmbedderFactory:
    """Tests for get_embedder factory function."""

    def test_openai_embedder_created(self):
        """EMBEDDER_TYPE=openai returns OpenAIEmbedder instance."""
        embedder = get_embedder(
            embedder_type="openai",
            openai_api_key="sk-test",
        )
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.api_key == "sk-test"

    def test_openai_embedder_case_insensitive(self):
        """EMBEDDER_TYPE is case-insensitive."""
        embedder = get_embedder(
            embedder_type="OPENAI",
            openai_api_key="sk-test",
        )
        assert isinstance(embedder, OpenAIEmbedder)

    def test_openai_embedder_without_key_raises(self):
        """Missing OpenAI API key raises EmbeddingError."""
        with pytest.raises(EmbeddingError) as exc_info:
            get_embedder(embedder_type="openai", openai_api_key=None)
        assert "OpenAI API key is required" in str(exc_info.value)

    def test_huggingface_embedder_created(self):
        """EMBEDDER_TYPE=huggingface returns HuggingFaceEmbedder instance."""
        embedder = get_embedder(
            embedder_type="huggingface",
            hf_api_token="hf-test",
        )
        assert isinstance(embedder, HuggingFaceEmbedder)
        assert embedder.api_token == "hf-test"

    def test_huggingface_embedder_default_model(self):
        """HuggingFace embedder uses default model if not specified."""
        embedder = get_embedder(
            embedder_type="huggingface",
            hf_api_token="hf-test",
        )
        assert embedder.model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_huggingface_embedder_custom_model(self):
        """HuggingFace embedder uses custom model when specified."""
        embedder = get_embedder(
            embedder_type="huggingface",
            hf_api_token="hf-test",
            hf_model="sentence-transformers/msmarco-MiniLM-L-6-v3",
        )
        assert embedder.model == "sentence-transformers/msmarco-MiniLM-L-6-v3"

    def test_huggingface_embedder_without_token_raises(self):
        """Missing HuggingFace token raises EmbeddingError."""
        with pytest.raises(EmbeddingError) as exc_info:
            get_embedder(embedder_type="huggingface", hf_api_token=None)
        assert "HuggingFace API token is required" in str(exc_info.value)

    def test_invalid_embedder_type_raises(self):
        """Invalid embedder_type raises EmbeddingError."""
        with pytest.raises(EmbeddingError) as exc_info:
            get_embedder(embedder_type="anthropic", openai_api_key="sk-test")
        assert "EMBEDDER_TYPE must be 'openai' or 'huggingface'" in str(exc_info.value)
