"""Tests for WikipediaIngester."""

import pytest
from unittest.mock import patch, MagicMock
from rhizome.corpus.wikipedia_ingester import WikipediaIngester, IngesterError
from rhizome.corpus.chunker import Chunker


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

    @patch("rhizome.corpus.wikipedia_ingester.requests.get")
    def test_fetch_article_success(self, mock_get):
        """_fetch_article returns extract from Wikipedia API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "12345": {
                        "pageid": 12345,
                        "title": "Modernism",
                        "extract": "Modernism is a philosophical movement."
                    }
                }
            }
        }
        mock_get.return_value = mock_response

        ingester = WikipediaIngester(domains=["Modernism"], seed_titles=["Modernism"])
        result = ingester._fetch_article("Modernism")

        assert result == "Modernism is a philosophical movement."
        # Verify correct params were passed
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["params"]["action"] == "query"
        assert call_kwargs.kwargs["params"]["prop"] == "extracts"

    @patch("rhizome.corpus.wikipedia_ingester.requests.get")
    def test_fetch_article_not_found(self, mock_get):
        """_fetch_article returns None for missing articles (no extract in response)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Wikipedia returns a page entry with missing 'extract' for nonexistent titles
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "-1": {
                        "ns": 0,
                        "title": "NonexistentArticle",
                        "missing": ""
                    }
                }
            }
        }
        mock_get.return_value = mock_response

        ingester = WikipediaIngester(domains=["Modernism"], seed_titles=["NonexistentArticle"])
        result = ingester._fetch_article("NonexistentArticle")

        assert result is None

    @patch("rhizome.corpus.wikipedia_ingester.requests.get")
    def test_fetch_article_api_error(self, mock_get):
        """_fetch_article raises IngesterError on HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        ingester = WikipediaIngester(domains=["Modernism"], seed_titles=["Modernism"])
        with pytest.raises(IngesterError) as exc_info:
            ingester._fetch_article("Modernism")
        assert "500" in str(exc_info.value)

    @patch("rhizome.corpus.wikipedia_ingester.requests.get")
    def test_ingest_yields_chunks_for_found_articles(self, mock_get):
        """ingest() yields chunks for articles found via Wikipedia API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "12345": {
                        "pageid": 12345,
                        "title": "Modernism",
                        "extract": "Modernism is a philosophical movement."
                    }
                }
            }
        }
        mock_get.return_value = mock_response

        ingester = WikipediaIngester(
            domains=["Modernism"],
            max_articles=10,
            seed_titles=["Modernism"],
            chunker=Chunker(min_chars=0),
        )

        chunks = list(ingester.ingest())
        assert len(chunks) >= 1
        assert chunks[0].article_title == "Modernism"
        assert "Modernism" in chunks[0].text

    @patch("rhizome.corpus.wikipedia_ingester.requests.get")
    def test_ingest_skips_missing_articles(self, mock_get):
        """Articles not found on Wikipedia are skipped with a warning."""
        # First call returns Modernism, second call returns missing
        mock_responses = [
            MagicMock(status_code=200, json=lambda: {
                "query": {
                    "pages": {
                        "12345": {
                            "pageid": 12345,
                            "title": "Modernism",
                            "extract": "Modernism is a philosophical movement."
                        }
                    }
                }
            }),
            MagicMock(status_code=200, json=lambda: {
                "query": {
                    "pages": {
                        "-1": {"title": "NonexistentArticle", "missing": ""}
                    }
                }
            }),
        ]
        mock_get.side_effect = mock_responses

        ingester = WikipediaIngester(
            domains=["Modernism"],
            max_articles=10,
            seed_titles=["Modernism", "NonexistentArticle"],
            chunker=Chunker(min_chars=0),
        )

        chunks = list(ingester.ingest())
        # Only Modernism chunk should be yielded
        assert all(c.article_title == "Modernism" for c in chunks)
