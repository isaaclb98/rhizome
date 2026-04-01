"""OpenAI API implementation of the Embedder interface."""

import os
import requests

from rhizome.embedder.base import Embedder


class OpenAIEmbedder(Embedder):
    """Embedder using OpenAI's embedding API.

    Args:
        api_key: OpenAI API key. Can also be set via OPENAI_API_KEY env var.
        model: Embedding model to use. Defaults to text-embedding-3-small.
               Alternatives: text-embedding-3-large, text-embedding-ada-002.
    """

    EMBEDDING_URL = "https://api.openai.com/v1/embeddings"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the OpenAI embedding API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If the API call fails.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self.model,
        }

        response = requests.post(
            self.EMBEDDING_URL,
            json=payload,
            headers=headers,
            timeout=60,
        )

        if response.status_code != 200:
            raise EmbeddingError(
                f"OpenAI API error: {response.status_code} {response.text}"
            )

        data = response.json()
        # text-embedding-3-small returns 1536-dim vectors
        return [item["embedding"] for item in data["data"]]

    def vector_size(self) -> int:
        """Return the embedding dimension for this model.

        Returns:
            Vector dimension (1536 for text-embedding-3-small,
            3072 for text-embedding-3-large, 1536 for ada-002).
        """
        sizes = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return sizes.get(self.model, 1536)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass
