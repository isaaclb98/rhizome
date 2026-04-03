"""Wikipedia article ingestion via PetScan + Wikipedia API."""

import os
import sys
import time
import requests
from typing import Iterator, Optional

from rhizome.corpus.chunker import Chunker, Chunk


PETSCAN_URL = "https://petscan.wmflabs.org/"
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


# Re-use the retry helper for PetScan
_petscan_with_retry = lambda url, params, timeout=60, max_retries=5: _http_with_retry(url, params, None, timeout, max_retries)


class WikipediaIngester:
    """Fetches Wikipedia articles and yields chunks for embedding.

    Discovery strategy: queries PetScan with category names to get the full
    set of articles in those Wikipedia categories (including subcategory members
    up to the configured depth). Article text is fetched on-demand via the
    Wikipedia API, one article at a time.

    Falls back to a seed list of article titles if provided.
    """

    def __init__(
        self,
        domains: str | list[str],
        max_articles: int = 500,
        seed_titles: list[str] | None = None,
        chunker: Chunker | None = None,
        depth: int = 1,
    ):
        self.domains = [domains] if isinstance(domains, str) else domains
        self.max_articles = max_articles
        self.seed_titles = seed_titles or []
        self.chunker = chunker or Chunker()
        self.depth = depth

    def ingest(self) -> Iterator[Chunk]:
        """Fetch articles and yield all chunks.

        Yields:
            Chunk objects for each article paragraph. Each chunk is tagged
            with the domain it was discovered under.

        Raises:
            IngesterError: If the Wikipedia API lookup fails or
                          PetScan is unreachable.
        """
        article_domains = self._discover_articles()
        for title, domain in list(article_domains.items())[: self.max_articles]:
            try:
                article_text = self._fetch_article(title)
                if article_text is None:
                    print(f"Warning: '{title}' not found on Wikipedia, skipping", file=sys.stderr)
                    continue
                chunks = self.chunker.chunk_article(
                    article_title=title,
                    article_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    article_text=article_text,
                    domain=domain,
                )
                for chunk in chunks:
                    yield chunk
            except IngesterError as e:
                print(f"Warning: failed to fetch '{title}': {e}", file=sys.stderr)
                continue

    def _discover_articles(self) -> dict[str, str]:
        """Discover article titles via PetScan category membership.

        Uses seed list if provided (returns each title tagged with the first domain),
        otherwise queries PetScan for each domain separately and tracks which domain
        each article was discovered under. If an article appears in multiple domains,
        it is assigned to the first domain that returned it.

        Returns:
            Dict mapping article title to domain name.
        """
        if self.seed_titles:
            first_domain = self.domains[0] if self.domains else "Unknown"
            return {title: first_domain for title in self.seed_titles}

        article_domains: dict[str, str] = {}

        for domain in self.domains:
            params = {
                "language": "en",
                "project": "wikipedia",
                "depth": self.depth,
                "categories": domain,
                "combination": "union",
                "format": "json",
                "doit": 1,
            }

            response = _http_with_retry(PETSCAN_URL, params=params)
            if response.status_code != 200:
                raise IngesterError(f"PetScan API error: {response.status_code} {response.text}")

            data = response.json()
            articles = data.get("*", [{}])[0].get("a", {}).get("*", [])
            for article in articles:
                title = article["title"].replace("_", " ")
                # Assign to first domain that returns this article
                if title not in article_domains:
                    article_domains[title] = domain

        return article_domains

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
        # Wikipedia API requires a real contact method in User-Agent.
        # Set WIKIPEDIA_USER_AGENT env var to override.
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
