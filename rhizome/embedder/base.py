"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import Protocol


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class Embedder(Protocol):
    """Interface for embedding providers.

    All embedding calls go through this interface, enabling:
    - Dependency injection and mocking in tests
    - Swap to different embedding providers (OpenAI, Anthropic, local) without
      changing call sites
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors. Each vector is a list of floats.
            The number of vectors equals len(texts).
            The dimension of each vector is consistent (e.g., 384 for all-MiniLM-L6-v2).
        """
        ...
