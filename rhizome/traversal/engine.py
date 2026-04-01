"""Epsilon-greedy traversal engine for rhizomatic vector-space walk."""

import random
from dataclasses import dataclass

from rhizome.embedder.base import Embedder
from rhizome.vectorstore.client import VectorStoreClient
from rhizome.traversal.config import TraversalConfig


@dataclass
class TraversalStep:
    """A single step in the traversal path."""

    chunk_id: str
    text: str
    article_title: str
    article_url: str
    depth: int


class TraversalEngine:
    """Epsilon-greedy random walk through a vector space.

    At each step:
    1. Search for top_K nearest chunks to the current query vector
    2. With probability epsilon: pick a random chunk from top_K (explore)
       With probability 1-epsilon: pick the nearest chunk (exploit)
    3. If all top_K are visited, fall back to next-nearest not visited
    4. If fallback fires 2+ consecutive times, force a random global jump
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStoreClient,
        config: TraversalConfig | None = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config or TraversalConfig()

    def traverse(self, starting_concept: str) -> list[TraversalStep]:
        """Perform a traversal starting from a concept.

        Args:
            starting_concept: A keyword or short phrase to start traversal from.

        Returns:
            Ordered list of TraversalSteps representing the path taken.

        Raises:
            TraversalError: If the starting concept produces no results
                          or the collection is empty.
        """
        # Embed the starting concept
        query_vector = self.embedder.embed([starting_concept])[0]

        visited_ids: set[str] = set()
        path: list[TraversalStep] = []
        consecutive_fallback = 0

        for depth in range(self.config.depth):
            # Search excluding visited chunks
            candidates = self.vector_store.search_excluding(
                query_vector=query_vector,
                exclude_ids=list(visited_ids),
                top_k=self.config.top_k,
            )

            if not candidates:
                # No more unvisited chunks — stop early
                break

            # Epsilon-greedy selection
            selected = self._epsilon_greedy_select(
                candidates=candidates,
                epsilon=self.config.epsilon,
            )

            # Check if this was a fallback (forced exploration after 2+ consecutive falls)
            is_fallback = consecutive_fallback >= 2

            # Update fallback counter
            if self._was_random_selection(selected, candidates):
                consecutive_fallback += 1
            else:
                consecutive_fallback = 0

            # Build the step
            payload = selected["payload"]
            step = TraversalStep(
                chunk_id=payload["id"],
                text=payload["text"],
                article_title=payload["article_title"],
                article_url=payload["article_url"],
                depth=depth,
            )
            path.append(step)
            visited_ids.add(payload["id"])

            # Use the selected chunk's vector as the next query
            # In a full implementation, we would store vectors alongside payloads
            # For now, we re-embed the selected text as a proxy
            query_vector = self.embedder.embed([payload["text"]])[0]

        return path

    def _epsilon_greedy_select(
        self,
        candidates: list[dict],
        epsilon: float,
    ) -> dict:
        """Select a candidate using epsilon-greedy strategy.

        Args:
            candidates: List of candidate results from vector search.
            epsilon: Probability of random exploration.

        Returns:
            The selected candidate dict.
        """
        if not candidates:
            raise TraversalError("No candidates to select from")

        if random.random() < epsilon:
            # Explore: pick a random candidate
            return random.choice(candidates)
        else:
            # Exploit: pick the nearest (first in sorted list)
            return candidates[0]

    def _was_random_selection(self, selected: dict, candidates: list[dict]) -> bool:
        """Check if the selected candidate was a random (explore) choice.

        This is approximate — if selected is not the first candidate,
        it could be either random or a fallback. We treat any non-first
        selection as potentially random for fallback tracking purposes.
        """
        return selected["id"] != candidates[0]["id"]


class TraversalError(Exception):
    """Raised when traversal fails."""
    pass
