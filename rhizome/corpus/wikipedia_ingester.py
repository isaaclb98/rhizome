"""Wikipedia article ingestion via PetScan + Wikipedia API."""

import os
import sys
import time
import requests
from typing import Iterator, Optional

from rhizome.corpus.chunker import Chunker, Chunk


PETSCAN_URL = "https://petscan.wmcloud.org/"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def _http_with_retry(
    url: str,
    params: dict,
    headers: dict | None = None,
    timeout: int = 30,
    max_retries: int = 5,
) -> requests.Response:
    """Fetch a URL with exponential backoff on transient HTTP errors."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code in (429, 503):
                raise requests.HTTPError(response=response)
            return response
        except requests.HTTPError as e:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) * 5
            print(f"HTTP {e.response.status_code}, retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) * 5
            print(f"Request failed ({type(e).__name__}), retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)


class WikipediaIngester:
    """Fetches Wikipedia articles and yields chunks for embedding.

    Queries PetScan with one or more Wikipedia categories (including subcategory
    members up to the configured depth). Article text is fetched on-demand via
    the Wikipedia API, one article at a time.
    """

    def __init__(
        self,
        categories: list[str],
        chunker: Chunker | None = None,
        depth: int = 1,
    ):
        self.categories = categories
        self.chunker = chunker or Chunker()
        self.depth = depth

    def ingest(self) -> Iterator[Chunk]:
        """Fetch articles and yield all chunks.

        Yields:
            Chunk objects for each article paragraph.

        Raises:
            IngesterError: If the Wikipedia API lookup fails or
                          PetScan is unreachable.
        """
        titles = self._discover_articles()
        for title in titles:
            try:
                article_text = self._fetch_article(title)
                if article_text is None:
                    print(f"Warning: '{title}' not found on Wikipedia, skipping", file=sys.stderr)
                    continue
                chunks = self.chunker.chunk_article(
                    article_title=title,
                    article_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    article_text=article_text,
                )
                for chunk in chunks:
                    yield chunk
            except IngesterError as e:
                print(f"Warning: failed to fetch '{title}': {e}", file=sys.stderr)
                continue

    def _discover_articles(self) -> list[str]:
        """Discover article titles via PetScan category membership.

        Runs a single PetScan query with all categories combined as a union.
        Articles appearing in multiple categories appear only once.

        Returns:
            List of article titles.
        """
        params = {
            "language": "en",
            "project": "wikipedia",
            "depth": self.depth,
            "categories": "\n".join(self.categories),
            "combination": "union",
            "format": "json",
            "doit": 1,
        }

        response = _http_with_retry(PETSCAN_URL, params=params)
        if response.status_code != 200:
            raise IngesterError(f"PetScan API error: {response.status_code} {response.text}")

        data = response.json()
        articles = data.get("*", [{}])[0].get("a", {}).get("*", [])
        return [article["title"].replace("_", " ") for article in articles]

    def _fetch_article(self, title: str) -> Optional[str]:
        """Fetch article text from the Wikipedia API.

        Makes an HTTP request to the Wikipedia API for each article with retry
        on transient errors. No dataset materialization — constant memory overhead.

        Returns:
            The article text, or None if the title does not exist.
        """
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }
        user_agent = os.environ.get(
            "WIKIPEDIA_USER_AGENT",
            "Rhizome/0.1.0 (Wikipedia Vector Corpus Builder; mailto:isaacbarney@hotmail.com)",
        )
        headers = {"User-Agent": user_agent}
        response = _http_with_retry(WIKIPEDIA_API, params=params, headers=headers)
        if response.status_code != 200:
            raise IngesterError(f"Wikipedia API error: {response.status_code}")

        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_data in pages.values():
            if "extract" in page_data:
                return page_data["extract"]
        return None


class IngesterError(Exception):
    """Raised when Wikipedia ingestion fails irrecoverably."""
    pass
