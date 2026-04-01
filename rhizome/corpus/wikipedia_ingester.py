"""Wikipedia article ingestion via PetScan + HuggingFace dataset."""

import sys
import time
import requests
from typing import Iterator, Optional

from rhizome.corpus.chunker import Chunker, Chunk


PETSCAN_URL = "https://petscan.wmflabs.org/"


def _petscan_with_retry(url: str, params: dict, timeout: int = 60, max_retries: int = 5) -> requests.Response:
    """Fetch PetScan with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code in (429, 503):
                raise requests.HTTPError(response=response)
            return response
        except requests.HTTPError as e:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) * 5
            print(f"PetScan returned {e.response.status_code}, retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)


class WikipediaIngester:
    """Fetches Wikipedia articles and yields chunks for embedding.

    Discovery strategy: queries PetScan with category names to get the full
    set of articles in those Wikipedia categories (including subcategory members
    up to the configured depth). Article text is then looked up by title
    from the HuggingFace wikimedia/wikipedia dataset snapshot.

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
        self._hf_stream = None

    def ingest(self) -> Iterator[Chunk]:
        """Fetch articles and yield all chunks.

        Yields:
            Chunk objects for each article paragraph.

        Raises:
            IngesterError: If the HuggingFace dataset lookup fails or
                          PetScan is unreachable.
        """
        titles = self._discover_articles()
        for title in titles[: self.max_articles]:
            try:
                article_text = self._fetch_article(title)
                if article_text is None:
                    print(f"Warning: '{title}' not found in HuggingFace dataset, skipping", file=sys.stderr)
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

        Uses seed list if provided, otherwise queries PetScan with the
        configured domains and returns the union of all matching articles.
        """
        if self.seed_titles:
            return self.seed_titles

        categories_str = "\r\n".join(self.domains)
        params = {
            "language": "en",
            "project": "wikipedia",
            "depth": self.depth,
            "categories": categories_str,
            "combination": "union",
            "format": "json",
            "doit": 1,
        }

        response = _petscan_with_retry(PETSCAN_URL, params=params)
        if response.status_code != 200:
            raise IngesterError(f"PetScan API error: {response.status_code} {response.text}")

        data = response.json()
        articles = data.get("*", [{}])[0].get("a", {}).get("*", [])
        return list({article["title"].replace("_", " ") for article in articles})

    def _fetch_article(self, title: str) -> Optional[str]:
        """Look up article text from the HuggingFace wikimedia/wikipedia dataset.

        The dataset is loaded once and materialized into memory for O(1) lookups.
        Wikipedia's 20231101.en snapshot is ~16GB. For large max_articles values
        this may require significant RAM.

        Returns:
            The article text, or None if the title is not in the dataset snapshot.
        """
        if self._hf_stream is None:
            from datasets import load_dataset

            ds = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",
                split="train",
                streaming=True,
            )
            # Materialize the full snapshot for O(1) lookups.
            # For very large corpora (50k+ articles), consider a machine with
            # more RAM or a sharded dataset approach.
            self._hf_stream = {example["title"]: example["text"] for example in ds}

        return self._hf_stream.get(title)


class IngesterError(Exception):
    """Raised when Wikipedia ingestion fails irrecoverably."""
    pass
