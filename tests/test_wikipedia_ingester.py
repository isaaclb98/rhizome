"""Tests for WikipediaIngester."""

import pytest
from unittest.mock import patch, MagicMock
from rhizome.corpus.wikipedia_ingester import WikipediaIngester, IngesterError
from rhizome.corpus.chunker import Chunker


class TestWikipediaIngester:
    """Tests for WikipediaIngester."""

    @patch("rhizome.corpus.wikipedia_ingester.requests.get")
    def test_discover_articles_success(self, mock_get):
        """_discover_articles parses PetScan JSON correctly and returns title list."""
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

        ingester = WikipediaIngester(categories=["Philosophy"])
        titles = ingester._discover_articles()

        assert "Modernism" in titles
        assert "Postmodernism" in titles

    @patch("rhizome.corpus.wikipedia_ingester.requests.get")
    def test_discover_articles_api_error(self, mock_get):
        """PetScan error raises IngesterError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        ingester = WikipediaIngester(categories=["Modernism"])
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

        ingester = WikipediaIngester(categories=["Modernism"])
        result = ingester._fetch_article("Modernism")

        assert result == "Modernism is a philosophical movement."
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["params"]["action"] == "query"
        assert call_kwargs.kwargs["params"]["prop"] == "extracts"

    @patch("rhizome.corpus.wikipedia_ingester.requests.get")
    def test_fetch_article_not_found(self, mock_get):
        """_fetch_article returns None for missing articles (no extract in response)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
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

        ingester = WikipediaIngester(categories=["Modernism"])
        result = ingester._fetch_article("NonexistentArticle")

        assert result is None

    @patch("rhizome.corpus.wikipedia_ingester.requests.get")
    def test_fetch_article_api_error(self, mock_get):
        """_fetch_article raises IngesterError on HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        ingester = WikipediaIngester(categories=["Modernism"])
        with pytest.raises(IngesterError) as exc_info:
            ingester._fetch_article("Modernism")
        assert "500" in str(exc_info.value)

    def test_ingest_yields_chunks_for_found_articles(self):
        """ingest() yields chunks for articles returned by _discover_articles."""
        ingester = WikipediaIngester(
            categories=["Modernism"],
            chunker=Chunker(min_chars=0),
        )
        # Patch _discover_articles to return a fixed list
        discovered_titles = ["Modernism"]

        # Patch _fetch_article to return article text
        def fake_fetch(title):
            if title == "Modernism":
                return "Modernism is a philosophical movement."
            return None

        with patch.object(ingester, "_discover_articles", return_value=discovered_titles):
            with patch.object(ingester, "_fetch_article", side_effect=fake_fetch):
                chunks = list(ingester.ingest())

        assert len(chunks) >= 1
        assert chunks[0].article_title == "Modernism"
        assert "Modernism" in chunks[0].text
