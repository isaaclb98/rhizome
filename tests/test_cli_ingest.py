"""Tests for CLI ingest command."""

import os
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from rhizome.cli.commands.ingest import ingest
from rhizome.corpus.wikipedia_ingester import IngesterError
from rhizome.embedder import EmbeddingError


class TestIngestCommand:
    """Tests for rhizome ingest CLI command."""

    @patch("rhizome.cli.commands.ingest.CollectionManager")
    @patch("rhizome.cli.commands.ingest.get_embedder")
    @patch("rhizome.cli.commands.ingest.WikipediaIngester")
    @patch("rhizome.cli.commands.ingest.get_config")
    def test_ingest_creates_collection_if_missing(
        self, mock_get_config, mock_ingester_cls, mock_get_embedder, mock_coll_mgr_cls
    ):
        """When collection doesn't exist, ingest creates it."""
        mock_cfg = MagicMock()
        mock_cfg.qdrant_url = "http://localhost:6333"
        mock_cfg.qdrant_collection = "test"
        mock_cfg.qdrant_api_key = None
        mock_cfg.embedder_type = "openai"
        mock_cfg.openai_api_key = "sk-test"
        mock_cfg.hf_api_token = None
        mock_cfg.hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_cfg.wikipedia_depth = 1
        mock_cfg.checkpoint_path = ".rhizome_checkpoints"
        mock_get_config.return_value = mock_cfg

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = False
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 1536]
        mock_embedder.vector_size.return_value = 1536
        mock_get_embedder.return_value = mock_embedder

        mock_ingester = MagicMock()
        mock_ingester.ingest.return_value = iter([
            MagicMock(
                text="Modernism is...",
                article_title="Modernism",
                article_url="https://en.wikipedia.org/wiki/Modernism",
            )
        ])
        mock_ingester_cls.return_value = mock_ingester

        runner = CliRunner()
        result = runner.invoke(ingest, ["--categories", "Modernism"])

        assert result.exit_code == 0
        mock_coll_mgr.create_collection.assert_called_once()

    @patch("rhizome.cli.commands.ingest.CollectionManager")
    @patch("rhizome.cli.commands.ingest.get_embedder")
    @patch("rhizome.cli.commands.ingest.WikipediaIngester")
    @patch("rhizome.cli.commands.ingest.get_config")
    def test_ingest_uses_existing_collection(
        self, mock_get_config, mock_ingester_cls, mock_get_embedder, mock_coll_mgr_cls
    ):
        """When collection exists, ingest uses it without recreating."""
        mock_cfg = MagicMock()
        mock_cfg.qdrant_url = "http://localhost:6333"
        mock_cfg.qdrant_collection = "test"
        mock_cfg.qdrant_api_key = None
        mock_cfg.embedder_type = "openai"
        mock_cfg.openai_api_key = "sk-test"
        mock_cfg.hf_api_token = None
        mock_cfg.hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_cfg.wikipedia_depth = 1
        mock_cfg.checkpoint_path = ".rhizome_checkpoints"
        mock_get_config.return_value = mock_cfg

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = True
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 1536]
        mock_embedder.vector_size.return_value = 1536
        mock_get_embedder.return_value = mock_embedder

        mock_ingester = MagicMock()
        mock_ingester.ingest.return_value = iter([
            MagicMock(
                text="Modernism is...",
                article_title="Modernism",
                article_url="https://en.wikipedia.org/wiki/Modernism",
            )
        ])
        mock_ingester_cls.return_value = mock_ingester

        runner = CliRunner()
        result = runner.invoke(ingest, ["--categories", "Modernism"])

        assert result.exit_code == 0
        mock_coll_mgr.create_collection.assert_not_called()

    @patch("rhizome.cli.commands.ingest.CollectionManager")
    @patch("rhizome.cli.commands.ingest.get_embedder")
    @patch("rhizome.cli.commands.ingest.WikipediaIngester")
    @patch("rhizome.cli.commands.ingest.get_config")
    def test_ingest_reports_article_count_correctly(
        self, mock_get_config, mock_ingester_cls, mock_get_embedder, mock_coll_mgr_cls
    ):
        """Article counter increments per unique title, not per batch."""
        mock_cfg = MagicMock()
        mock_cfg.qdrant_url = "http://localhost:6333"
        mock_cfg.qdrant_collection = "test"
        mock_cfg.qdrant_api_key = None
        mock_cfg.embedder_type = "openai"
        mock_cfg.openai_api_key = "sk-test"
        mock_cfg.hf_api_token = None
        mock_cfg.hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_cfg.wikipedia_depth = 1
        mock_cfg.checkpoint_path = ".rhizome_checkpoints"
        mock_get_config.return_value = mock_cfg

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = True
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 1536]
        mock_embedder.vector_size.return_value = 1536
        mock_get_embedder.return_value = mock_embedder

        # Simulate 2 articles with 60 chunks each (2 batches of 50)
        mock_chunks = []
        for title in ["Article A", "Article B"]:
            for i in range(60):
                mock_chunks.append(
                    MagicMock(
                        text=f"Chunk {i} of {title}",
                        article_title=title,
                        article_url=f"https://en.wikipedia.org/wiki/{title}",
                    )
                )
        mock_ingester = MagicMock()
        mock_ingester.ingest.return_value = iter(mock_chunks)
        mock_ingester._discover_articles.return_value = ["Article A", "Article B"]
        mock_ingester_cls.return_value = mock_ingester

        runner = CliRunner()
        result = runner.invoke(ingest, ["--categories", "Modernism"])

        assert result.exit_code == 0
        # Should report 2 articles, not 2 batches
        assert "2 articles" in result.output

    @patch("rhizome.cli.commands.ingest.CollectionManager")
    @patch("rhizome.cli.commands.ingest.get_embedder")
    @patch("rhizome.cli.commands.ingest.WikipediaIngester")
    @patch("rhizome.cli.commands.ingest.get_config")
    def test_ingest_missing_openai_key_aborts(
        self, mock_get_config, mock_ingester_cls, mock_get_embedder, mock_coll_mgr_cls
    ):
        """Missing OpenAI API key causes EmbeddingError which causes Abort."""
        mock_cfg = MagicMock()
        mock_cfg.qdrant_url = "http://localhost:6333"
        mock_cfg.qdrant_collection = "test"
        mock_cfg.qdrant_api_key = None
        mock_cfg.embedder_type = "openai"
        mock_cfg.openai_api_key = None  # Missing key
        mock_cfg.hf_api_token = None
        mock_cfg.hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_cfg.wikipedia_depth = 1
        mock_cfg.checkpoint_path = ".rhizome_checkpoints"
        mock_get_config.return_value = mock_cfg

        mock_get_embedder.side_effect = EmbeddingError(
            "OpenAI API key is required when EMBEDDER_TYPE=openai."
        )

        runner = CliRunner()
        result = runner.invoke(ingest, ["--categories", "Modernism"])

        assert result.exit_code != 0
