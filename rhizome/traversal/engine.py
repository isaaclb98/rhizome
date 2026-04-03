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
    domain: str
    depth: int
    similarity: float
    forced_jump: bool
    # All top_k candidates considered at this step (selected is first)
    candidates: list[dict]


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
        in_forced_jump = False

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

            selected: dict

            # Forced global jump: 2+ consecutive fallbacks means we're stuck
            if in_forced_jump or consecutive_fallback >= 2:
                # Do a forced global jump — broad search then random pick
                # Pass with_vector=False since we only use IDs for random selection
                broad = self.vector_store.search(
                    query_vector=query_vector,
                    top_k=50,
                    with_vector=False,
                )
                remaining = [c for c in broad if c["id"] not in visited_ids]
                if remaining:
                    selected = random.choice(remaining)
                else:
                    break  # No unvisited chunks at all
                in_forced_jump = True
                consecutive_fallback = 0
            else:
                # Normal epsilon-greedy selection
                selected, explore_fired = self._epsilon_greedy_select(
                    candidates=candidates,
                    epsilon=self.config.epsilon,
                )
                # Track fallback: we fell back when the best candidate was already visited
                # (epsilon exploration is tracked separately)
                if not explore_fired and selected["id"] != candidates[0]["id"]:
                    consecutive_fallback += 1
                else:
                    consecutive_fallback = 0

            # Build the step — include all candidates so the UI can draw kNN edges
            payload = selected["payload"]
            step = TraversalStep(
                chunk_id=payload["id"],
                text=payload["text"],
                article_title=payload["article_title"],
                article_url=payload["article_url"],
                domain=payload.get("domain", "Unknown"),
                depth=depth,
                similarity=float(selected["score"]),
                forced_jump=in_forced_jump,
                candidates=candidates,
            )
            path.append(step)
            visited_ids.add(payload["id"])

            # After a forced jump, next step is normal traversal
            in_forced_jump = False

            # Use the selected chunk's stored vector as the next query.
            # Fall back to re-embedding only if the stored vector is unavailable.
            stored_vector = selected.get("vector")
            if stored_vector is not None:
                query_vector = stored_vector
            else:
                query_vector = self.embedder.embed([payload["text"]])[0]

        return path

    def _epsilon_greedy_select(
        self,
        candidates: list[dict],
        epsilon: float,
    ) -> tuple[dict, bool]:
        """Select a candidate using epsilon-greedy strategy.

        Args:
            candidates: List of candidate results from vector search.
            epsilon: Probability of random exploration.

        Returns:
            A tuple of (selected candidate dict, explore_fired bool).
            explore_fired is True when epsilon triggered random exploration.
        """
        if not candidates:
            raise TraversalError("No candidates to select from")

        if random.random() < epsilon:
            # Explore: pick a random candidate
            return random.choice(candidates), True
        else:
            # Exploit: pick the nearest (first in sorted list)
            return candidates[0], False


class TraversalError(Exception):
    """Raised when traversal fails."""
    pass
