"""Tests for WikipediaIngester."""

import pytest
from unittest.mock import patch, MagicMock
from rhizome.corpus.wikipedia_ingester import WikipediaIngester, IngesterError


class TestWikipediaIngester:
    """Tests for WikipediaIngester."""

    def test_ingest_with_seed_titles(self):
        """Seed titles bypass PetScan discovery."""
        ingester = WikipediaIngester(
            domains=["Modernism"],
            max_articles=10,
            seed_titles=["Article A", "Article B"],
        )
        # If seed_titles are provided, _discover_articles returns them directly
        titles = ingester._discover_articles()
        assert titles == ["Article A", "Article B"]

    @patch("rhizome.corpus.wikipedia_ingester.requests.get")
    def test_discover_articles_success(self, mock_get):
        """_discover_articles parses PetScan JSON correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "*": [
                {
                    "a": {
                        "*": [
                            {"title": "Modernism"},
                            {"title": "Postmodernism"},
                        ]
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        ingester = WikipediaIngester(domains=["Philosophy"], max_articles=50)
        titles = ingester._discover_articles()

        assert "Modernism" in titles
        assert "Postmodernism" in titles
        # Deduplication
        assert len(titles) == len(set(titles))

    @patch("rhizome.corpus.wikipedia_ingester.requests.get")
    def test_discover_articles_api_error(self, mock_get):
        """PetScan error raises IngesterError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        ingester = WikipediaIngester(domains=["Modernism"])
        with pytest.raises(IngesterError) as exc_info:
            ingester._discover_articles()
        assert "500" in str(exc_info.value)

    def test_fetch_article_materializes_stream(self):
        """_fetch_article caches the HF stream as a dict on first call."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = lambda self: iter(
            [{"title": "Modernism", "text": "Modernism is..."}, {"title": "Postmodernism", "text": "Postmodernism is..."}]
        )

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_ds
            ingester = WikipediaIngester(domains=["Modernism"], seed_titles=["Modernism", "Postmodernism"])

            # First call — stream should be materialized
            result1 = ingester._fetch_article("Modernism")
            assert result1 == "Modernism is..."

            # Verify load_dataset was called once
            mock_load.assert_called_once()
            # Verify stream was converted to dict
            assert isinstance(ingester._hf_stream, dict)

            # Second call — should use cached dict, not re-iterate
            result2 = ingester._fetch_article("Postmodernism")
            assert result2 == "Postmodernism is..."
            # load_dataset should NOT be called again
            mock_load.assert_called_once()

    def test_fetch_article_not_found(self):
        """_fetch_article returns None for missing articles."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = lambda self: iter([{"title": "Modernism", "text": "Modernism is..."}])

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_ds
            ingester = WikipediaIngester(domains=["Modernism"], seed_titles=["Modernism"])
            ingester._fetch_article("Modernism")  # materialize

            result = ingester._fetch_article("Nonexistent Article")
            assert result is None

    @patch("datasets.load_dataset")
    def test_ingest_yields_chunks_for_found_articles(self, mock_load):
        """ingest() yields chunks for articles found in HF dataset."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = lambda self: iter([{"title": "Modernism", "text": "Modernism is a movement."}])

        mock_load.return_value = mock_ds
        ingester = WikipediaIngester(
            domains=["Modernism"],
            max_articles=10,
            seed_titles=["Modernism"],
        )

        chunks = list(ingester.ingest())
        assert len(chunks) >= 1
        assert chunks[0].article_title == "Modernism"
        assert "Modernism" in chunks[0].text

    @patch("datasets.load_dataset")
    def test_ingest_skips_missing_articles(self, mock_load, capsys):
        """Articles not in HF dataset are skipped with a warning."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = lambda self: iter([{"title": "Modernism", "text": "Modernism is a movement."}])

        mock_load.return_value = mock_ds
        ingester = WikipediaIngester(
            domains=["Modernism"],
            max_articles=10,
            seed_titles=["Modernism", "NonexistentArticle"],
        )

        chunks = list(ingester.ingest())
        # Only Modernism chunk should be yielded
        assert all(c.article_title == "Modernism" for c in chunks)
