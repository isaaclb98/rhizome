"""Tests for CLI traverse command."""

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from rhizome.cli.commands.traverse import traverse
from rhizome.traversal.engine import TraversalError


class TestTraverseCommand:
    """Tests for rhizome traverse CLI command."""

    @patch("rhizome.cli.commands.traverse.TraversalEngine")
    @patch("rhizome.cli.commands.traverse.VectorStoreClient")
    @patch("rhizome.cli.commands.traverse.CollectionManager")
    @patch("rhizome.cli.commands.traverse.get_embedder")
    @patch("rhizome.cli.commands.traverse.get_config")
    def test_traverse_collection_not_found_aborts(
        self, mock_get_config, mock_get_embedder, mock_coll_mgr_cls, mock_vec_store_cls, mock_engine_cls
    ):
        """Missing collection causes Abort with helpful message."""
        mock_cfg = MagicMock()
        mock_cfg.qdrant_url = "http://localhost:6333"
        mock_cfg.qdrant_collection = "test"
        mock_cfg.qdrant_api_key = None
        mock_cfg.embedder_type = "openai"
        mock_cfg.openai_api_key = "sk-test"
        mock_cfg.hf_api_token = None
        mock_cfg.hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_cfg.default_depth = 8
        mock_cfg.epsilon = 0.1
        mock_get_config.return_value = mock_cfg

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = False
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        runner = CliRunner()
        result = runner.invoke(traverse, ["modernism"])

        assert result.exit_code != 0
        assert "not found" in result.output

    @patch("rhizome.cli.commands.traverse.TraversalEngine")
    @patch("rhizome.cli.commands.traverse.VectorStoreClient")
    @patch("rhizome.cli.commands.traverse.CollectionManager")
    @patch("rhizome.cli.commands.traverse.get_embedder")
    @patch("rhizome.cli.commands.traverse.get_config")
    def test_traverse_error_raises_abort(
        self, mock_get_config, mock_get_embedder, mock_coll_mgr_cls, mock_vec_store_cls, mock_engine_cls
    ):
        """TraversalError is caught and causes Abort."""
        mock_cfg = MagicMock()
        mock_cfg.qdrant_url = "http://localhost:6333"
        mock_cfg.qdrant_collection = "test"
        mock_cfg.qdrant_api_key = None
        mock_cfg.embedder_type = "openai"
        mock_cfg.openai_api_key = "sk-test"
        mock_cfg.hf_api_token = None
        mock_cfg.hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_cfg.default_depth = 8
        mock_cfg.epsilon = 0.1
        mock_get_config.return_value = mock_cfg

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = True
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        mock_embedder = MagicMock()
        mock_embedder.vector_size.return_value = 1536
        mock_get_embedder.return_value = mock_embedder

        mock_engine = MagicMock()
        mock_engine.traverse.side_effect = TraversalError("Empty collection")
        mock_engine_cls.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(traverse, ["modernism"])

        assert result.exit_code != 0
        assert "Traversal error" in result.output

    @patch("rhizome.cli.commands.traverse.TraversalEngine")
    @patch("rhizome.cli.commands.traverse.VectorStoreClient")
    @patch("rhizome.cli.commands.traverse.CollectionManager")
    @patch("rhizome.cli.commands.traverse.get_embedder")
    @patch("rhizome.cli.commands.traverse.get_config")
    def test_traverse_empty_path_aborts(
        self, mock_get_config, mock_get_embedder, mock_coll_mgr_cls, mock_vec_store_cls, mock_engine_cls
    ):
        """Empty path causes Abort with helpful message."""
        mock_cfg = MagicMock()
        mock_cfg.qdrant_url = "http://localhost:6333"
        mock_cfg.qdrant_collection = "test"
        mock_cfg.qdrant_api_key = None
        mock_cfg.embedder_type = "openai"
        mock_cfg.openai_api_key = "sk-test"
        mock_cfg.hf_api_token = None
        mock_cfg.hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_cfg.default_depth = 8
        mock_cfg.epsilon = 0.1
        mock_get_config.return_value = mock_cfg

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = True
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        mock_embedder = MagicMock()
        mock_embedder.vector_size.return_value = 1536
        mock_get_embedder.return_value = mock_embedder

        mock_engine = MagicMock()
        mock_engine.traverse.return_value = []  # Empty path
        mock_engine_cls.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(traverse, ["nonexistent concept"])

        assert result.exit_code != 0
        assert "too small" in result.output or "no path" in result.output

    @patch("rhizome.cli.commands.traverse.TraversalEngine")
    @patch("rhizome.cli.commands.traverse.VectorStoreClient")
    @patch("rhizome.cli.commands.traverse.CollectionManager")
    @patch("rhizome.cli.commands.traverse.get_embedder")
    @patch("rhizome.cli.commands.traverse.get_config")
    def test_traverse_writes_to_output_file(
        self, mock_get_config, mock_get_embedder, mock_coll_mgr_cls, mock_vec_store_cls, mock_engine_cls, tmp_path
    ):
        """-o flag writes markdown to the specified file."""
        mock_cfg = MagicMock()
        mock_cfg.qdrant_url = "http://localhost:6333"
        mock_cfg.qdrant_collection = "test"
        mock_cfg.qdrant_api_key = None
        mock_cfg.embedder_type = "openai"
        mock_cfg.openai_api_key = "sk-test"
        mock_cfg.hf_api_token = None
        mock_cfg.hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_cfg.default_depth = 8
        mock_cfg.epsilon = 0.1
        mock_get_config.return_value = mock_cfg

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = True
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        mock_embedder = MagicMock()
        mock_embedder.vector_size.return_value = 1536
        mock_get_embedder.return_value = mock_embedder

        mock_step = MagicMock()
        mock_step.text = "Modernism is a movement."
        mock_step.article_title = "Modernism"
        mock_step.article_url = "https://en.wikipedia.org/wiki/Modernism"

        mock_engine = MagicMock()
        mock_engine.traverse.return_value = [mock_step]
        mock_engine_cls.return_value = mock_engine

        output = tmp_path / "output.md"
        runner = CliRunner()
        result = runner.invoke(
            traverse, ["modernism", "-o", str(output)]
        )

        assert result.exit_code == 0
        assert output.read_text() != ""
