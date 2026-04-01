"""Wikipedia article ingestion via the Wikipedia API."""

import requests
import time
from typing import Iterator

from rhizome.corpus.chunker import Chunker, Chunk


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


class WikipediaIngester:
    """Fetches Wikipedia articles and yields chunks for embedding.

    Discovery strategy: takes a domain name, uses Wikipedia search API
    to find relevant articles, then fetches full article text.

    For the prototype, pass a seed list of article titles to guarantee
    reliable content. Falls back to search-based discovery if no seed list.
    """

    def __init__(
        self,
        domain: str,
        max_articles: int = 500,
        seed_titles: list[str] | None = None,
        chunker: Chunker | None = None,
    ):
        self.domain = domain
        self.max_articles = max_articles
        self.seed_titles = seed_titles or []
        self.chunker = chunker or Chunker()

    def ingest(self) -> Iterator[Chunk]:
        """Fetch articles and yield all chunks.

        Yields:
            Chunk objects for each article paragraph.

        Raises:
            IngesterError: If Wikipedia API returns an unexpected error or
                          rate limiting exhausts retries.
        """
        titles = self._discover_articles()
        for title in titles[:self.max_articles]:
            try:
                chunks = self._fetch_article(title)
                for chunk in chunks:
                    yield chunk
            except IngesterError as e:
                # Log and continue — partial corpus is acceptable
                print(f"Warning: failed to fetch '{title}': {e}")
                continue

    def _discover_articles(self) -> list[str]:
        """Discover article titles for the domain.

        Uses seed list if provided, otherwise falls back to Wikipedia search API.
        """
        if self.seed_titles:
            return self.seed_titles

        # Search Wikipedia for articles matching the domain
        search_url = WIKIPEDIA_API
        params = {
            "action": "query",
            "list": "search",
            "srsearch": self.domain,
            "srlimit": self.max_articles,
            "format": "json",
        }

        response = self._http_get(search_url, params)
        data = response.json()
        results = data.get("query", {}).get("search", [])
        return [r["title"] for r in results]

    def _fetch_article(self, title: str) -> list[Chunk]:
        """Fetch a single article's full text and chunk it."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }

        response = self._http_get(WIKIPEDIA_API, params)
        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        for page_id, page_data in pages.items():
            if page_id == "-1":  # Page not found
                raise IngesterError(f"Article not found: {title}")

            article_text = page_data.get("extract", "")
            if not article_text:
                raise IngesterError(f"No content for article: {title}")

            return self.chunker.chunk_article(
                article_title=title,
                article_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                article_text=article_text,
            )

        raise IngesterError(f"Unexpected response for article: {title}")

    def _http_get(self, url: str, params: dict, retries: int = 3) -> requests.Response:
        """GET with exponential backoff for rate limiting."""
        backoff = 1.0
        for attempt in range(retries):
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response
            if response.status_code == 429:  # Rate limited
                time.sleep(backoff)
                backoff *= 2
                continue
            raise IngesterError(f"HTTP {response.status_code}: {response.text}")

        raise IngesterError(f"Rate limited after {retries} retries")


class IngesterError(Exception):
    """Raised when Wikipedia ingestion fails irrecoverably."""
    pass
