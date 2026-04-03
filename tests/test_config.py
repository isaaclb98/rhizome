"""Tests for the Pydantic configuration module."""

import os
import pytest
from unittest.mock import patch
from pydantic import ValidationError
from rhizome.config import RhizomeConfig, get_config


class TestRhizomeConfig:
    """Tests for RhizomeConfig."""

    def test_defaults(self):
        """Default values are applied when env vars are not set."""
        cfg = RhizomeConfig(
            QDRANT_COLLECTION="test-collection",
            EMBEDDER_TYPE="openai",
            OPENAI_API_KEY="sk-test",
        )
        assert cfg.qdrant_url == "http://localhost:6333"
        assert cfg.qdrant_collection == "test-collection"
        assert cfg.default_depth == 8
        assert cfg.epsilon == 0.1
        assert cfg.wikipedia_domains == ["Modernism", "Postmodernism", "Critical theory"]

    def test_embedder_type_normalized(self):
        """embedder_type is normalized to lowercase."""
        cfg = RhizomeConfig(
            QDRANT_COLLECTION="test",
            EMBEDDER_TYPE="OPENAI",
            OPENAI_API_KEY="sk-test",
        )
        assert cfg.embedder_type == "openai"

    def test_embedder_type_invalid_raises(self):
        """Invalid embedder_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RhizomeConfig(
                QDRANT_COLLECTION="test",
                EMBEDDER_TYPE="invalid",
                OPENAI_API_KEY="sk-test",
            )
        assert "EMBEDDER_TYPE must be 'openai' or 'huggingface'" in str(exc_info.value)

    def test_wikipedia_domains_comma_separated(self):
        """Comma-separated domain string is parsed into list."""
        cfg = RhizomeConfig(
            QDRANT_COLLECTION="test",
            EMBEDDER_TYPE="openai",
            OPENAI_API_KEY="sk-test",
            WIKIPEDIA_DOMAINS="Modernism, Postmodernism, Critical theory",
        )
        assert cfg.wikipedia_domains == ["Modernism", "Postmodernism", "Critical theory"]

    def test_resolve_env_var_syntax(self):
        """${VAR} syntax resolves to environment variable value."""
        os.environ["TEST_SECRET"] = "secret-value"
        try:
            cfg = RhizomeConfig(
                QDRANT_COLLECTION="test",
                EMBEDDER_TYPE="openai",
                OPENAI_API_KEY="${TEST_SECRET}",
            )
            assert cfg.openai_api_key == "secret-value"
        finally:
            del os.environ["TEST_SECRET"]

    def test_require_openai_key_raises(self):
        """require_openai_key raises when key is not set."""
        cfg = RhizomeConfig(
            QDRANT_COLLECTION="test",
            EMBEDDER_TYPE="openai",
            openai_api_key=None,
        )
        with pytest.raises(ValueError) as exc_info:
            cfg.require_openai_key()
        assert "OPENAI_API_KEY environment variable is not set" in str(exc_info.value)

    def test_require_hf_token_raises(self):
        """require_hf_token raises when token is not set."""
        cfg = RhizomeConfig(
            QDRANT_COLLECTION="test",
            EMBEDDER_TYPE="huggingface",
            hf_api_token=None,
        )
        with pytest.raises(ValueError) as exc_info:
            cfg.require_hf_token()
        assert "HF_API_TOKEN environment variable is not set" in str(exc_info.value)

    def test_get_config_returns_singleton(self):
        """get_config returns the same instance on subsequent calls."""
        # Clear the global _config
        import rhizome.config
        rhizome.config._config = None

        with patch.dict(os.environ, {
            "QDRANT_COLLECTION": "test",
            "EMBEDDER_TYPE": "openai",
            "OPENAI_API_KEY": "sk-test",
        }):
            cfg1 = get_config()
            cfg2 = get_config()
            assert cfg1 is cfg2
