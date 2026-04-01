"""Tests for CLI traverse command."""

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from rhizome.cli.commands.traverse import traverse
from rhizome.traversal.engine import TraversalError


class TestTraverseCommand:
    """Tests for rhizome traverse CLI command."""

    # Note: OPENAI_API_KEY missing/empty → caught as OpenAI API auth error at
    # embed time (traverse), not at startup. Covered by test_ingest_missing_openai_key_aborts.

    @patch("rhizome.cli.commands.traverse.TraversalEngine")
    @patch("rhizome.cli.commands.traverse.CollectionManager")
    @patch("rhizome.cli.commands.traverse.OpenAIEmbedder")
    def test_traverse_collection_not_found_aborts(
        self, mock_embedder_cls, mock_coll_mgr_cls, mock_engine_cls, tmp_path
    ):
        """Missing collection causes Abort with helpful message."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "vectorstore:\n  url: 'http://localhost:6333'\n  collection: 'test'\n"
            "openai:\n  api_key: 'sk-test'\n"
            "traversal:\n  depth: 8\n"
        )

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = False
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        runner = CliRunner()
        result = runner.invoke(traverse, ["modernism", "--config", str(cfg)])

        assert result.exit_code != 0
        assert "not found" in result.output

    @patch("rhizome.cli.commands.traverse.TraversalEngine")
    @patch("rhizome.cli.commands.traverse.CollectionManager")
    @patch("rhizome.cli.commands.traverse.OpenAIEmbedder")
    def test_traverse_error_raises_abort(
        self, mock_embedder_cls, mock_coll_mgr_cls, mock_engine_cls, tmp_path
    ):
        """TraversalError is caught and causes Abort."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "vectorstore:\n  url: 'http://localhost:6333'\n  collection: 'test'\n"
            "openai:\n  api_key: 'sk-test'\n"
            "traversal:\n  depth: 8\n"
        )

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = True
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        mock_engine = MagicMock()
        mock_engine.traverse.side_effect = TraversalError("Empty collection")
        mock_engine_cls.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(traverse, ["modernism", "--config", str(cfg)])

        assert result.exit_code != 0
        assert "Traversal error" in result.output

    @patch("rhizome.cli.commands.traverse.TraversalEngine")
    @patch("rhizome.cli.commands.traverse.CollectionManager")
    @patch("rhizome.cli.commands.traverse.OpenAIEmbedder")
    def test_traverse_empty_path_aborts(
        self, mock_embedder_cls, mock_coll_mgr_cls, mock_engine_cls, tmp_path
    ):
        """Empty path causes Abort with helpful message."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "vectorstore:\n  url: 'http://localhost:6333'\n  collection: 'test'\n"
            "openai:\n  api_key: 'sk-test'\n"
            "traversal:\n  depth: 8\n"
        )

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = True
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        mock_engine = MagicMock()
        mock_engine.traverse.return_value = []  # Empty path
        mock_engine_cls.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(traverse, ["nonexistent concept", "--config", str(cfg)])

        assert result.exit_code != 0
        assert "too small" in result.output or "no path" in result.output

    @patch("rhizome.cli.commands.traverse.TraversalEngine")
    @patch("rhizome.cli.commands.traverse.CollectionManager")
    @patch("rhizome.cli.commands.traverse.OpenAIEmbedder")
    def test_traverse_writes_to_output_file(
        self, mock_embedder_cls, mock_coll_mgr_cls, mock_engine_cls, tmp_path
    ):
        """-o flag writes markdown to the specified file."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "vectorstore:\n  url: 'http://localhost:6333'\n  collection: 'test'\n"
            "openai:\n  api_key: 'sk-test'\n"
            "traversal:\n  depth: 8\n"
        )
        output = tmp_path / "output.md"

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = True
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        mock_step = MagicMock()
        mock_step.text = "Modernism is a movement."
        mock_step.article_title = "Modernism"
        mock_step.article_url = "https://en.wikipedia.org/wiki/Modernism"

        mock_engine = MagicMock()
        mock_engine.traverse.return_value = [mock_step]
        mock_engine_cls.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(
            traverse, ["modernism", "--config", str(cfg), "-o", str(output)]
        )

        assert result.exit_code == 0
        assert output.read_text() != ""
