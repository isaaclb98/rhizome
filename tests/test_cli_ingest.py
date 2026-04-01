"""Tests for CLI ingest command."""

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from rhizome.cli.commands.ingest import ingest
from rhizome.corpus.wikipedia_ingester import IngesterError


class TestIngestCommand:
    """Tests for rhizome ingest CLI command."""

    @patch("rhizome.cli.commands.ingest.CollectionManager")
    @patch("rhizome.cli.commands.ingest.OpenAIEmbedder")
    @patch("rhizome.cli.commands.ingest.WikipediaIngester")
    def test_ingest_creates_collection_if_missing(
        self, mock_ingester_cls, mock_embedder_cls, mock_coll_mgr_cls, tmp_path
    ):
        """When collection doesn't exist, ingest creates it."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "vectorstore:\n  url: 'http://localhost:6333'\n  collection: 'test'\n  vector_size: 1536\n"
            "corpus:\n  domains: ['Modernism']\n  max_articles: 10\n"
            "openai:\n  api_key: 'sk-test'\n"
            "traversal:\n  depth: 8\n"
        )

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = False
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 1536]
        mock_embedder_cls.return_value = mock_embedder

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
        result = runner.invoke(ingest, ["--config", str(cfg)])

        assert result.exit_code == 0
        mock_coll_mgr.create_collection.assert_called_once()

    @patch("rhizome.cli.commands.ingest.CollectionManager")
    @patch("rhizome.cli.commands.ingest.OpenAIEmbedder")
    @patch("rhizome.cli.commands.ingest.WikipediaIngester")
    def test_ingest_uses_existing_collection(
        self, mock_ingester_cls, mock_embedder_cls, mock_coll_mgr_cls, tmp_path
    ):
        """When collection exists, ingest uses it without recreating."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "vectorstore:\n  url: 'http://localhost:6333'\n  collection: 'test'\n  vector_size: 1536\n"
            "corpus:\n  domains: ['Modernism']\n  max_articles: 10\n"
            "openai:\n  api_key: 'sk-test'\n"
            "traversal:\n  depth: 8\n"
        )

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = True
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 1536]
        mock_embedder_cls.return_value = mock_embedder

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
        result = runner.invoke(ingest, ["--config", str(cfg)])

        assert result.exit_code == 0
        mock_coll_mgr.create_collection.assert_not_called()

    def test_ingest_missing_openai_key_aborts(self, tmp_path, monkeypatch):
        """Missing OPENAI_API_KEY causes Abort with error message."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "vectorstore:\n  url: 'http://localhost:6333'\n  collection: 'test'\n"
            "corpus:\n  domains: ['Modernism']\n"
            "openai:\n  api_key: '${NONEXISTENT_KEY}'\n"
            "traversal: {}\n"
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        runner = CliRunner()
        result = runner.invoke(ingest, ["--config", str(cfg)])

        assert result.exit_code != 0
        assert "OPENAI_API_KEY" in result.output

    @patch("rhizome.cli.commands.ingest.CollectionManager")
    @patch("rhizome.cli.commands.ingest.OpenAIEmbedder")
    @patch("rhizome.cli.commands.ingest.WikipediaIngester")
    def test_ingest_reports_article_count_correctly(
        self, mock_ingester_cls, mock_embedder_cls, mock_coll_mgr_cls, tmp_path
    ):
        """Article counter increments per unique title, not per batch."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "vectorstore:\n  url: 'http://localhost:6333'\n  collection: 'test'\n  vector_size: 1536\n"
            "corpus:\n  domains: ['Modernism']\n  max_articles: 10\n"
            "openai:\n  api_key: 'sk-test'\n"
            "traversal:\n  depth: 8\n"
        )

        mock_coll_mgr = MagicMock()
        mock_coll_mgr.collection_exists.return_value = True
        mock_coll_mgr_cls.return_value = mock_coll_mgr

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 1536]
        mock_embedder_cls.return_value = mock_embedder

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
        mock_ingester_cls.return_value = mock_ingester

        runner = CliRunner()
        result = runner.invoke(ingest, ["--config", str(cfg)])

        assert result.exit_code == 0
        # Should report 2 articles, not 2 batches
        assert "2 articles" in result.output
