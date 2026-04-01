"""HuggingFace Inference API implementation of the Embedder interface."""

import requests

from rhizome.embedder.base import Embedder


class HuggingFaceEmbedder(Embedder):
    """Embedder using HuggingFace's Inference API.

    Args:
        api_token: HuggingFace API token. Can also be set via HF_API_TOKEN env var.
        model: Model to use. Defaults to sentence-transformers/all-MiniLM-L6-v2.
    """

    def __init__(
        self,
        api_token: str | None = None,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        import os
        self.api_token = api_token or os.environ.get("HF_API_TOKEN")
        self.model = model
        self._endpoint = "https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using HuggingFace Inference API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If the API call fails.
        """
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        url = self._endpoint.format(model=self.model)

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code != 200:
            raise EmbeddingError(f"HuggingFace API error: {response.status_code} {response.text}")

        return response.json()

    def vector_size(self) -> int:
        """Return the expected vector dimension for this model.

        Returns:
            Vector dimension. Defaults to 384 for all-MiniLM-L6-v2.
        """
        return 384  # all-MiniLM-L6-v2 outputs 384-dimensional vectors


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass
