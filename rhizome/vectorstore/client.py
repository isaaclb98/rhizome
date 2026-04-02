"""Qdrant client wrapper for vector search operations."""

from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams, Filter, FieldCondition, MatchValue

from rhizome.corpus.chunker import Chunk


class VectorStoreClient:
    """Wrapper around Qdrant client for vector search operations."""

    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None, collection_name: str = "modernity-v1"):
        # When using HTTPS, Qdrant Cloud serves on port 443 (not the default 6333)
        port = 443 if url.startswith("https://") else None
        self.client = QdrantClient(url=url, api_key=api_key, port=port)
        self.collection_name = collection_name

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        query_filter: Filter | None = None,
    ) -> list[dict]:
        """Search for the nearest chunks to a query vector.

        Args:
            query_vector: The query embedding vector.
            top_k: Number of results to return.
            query_filter: Optional Qdrant filter.

        Returns:
            List of dicts with 'id', 'score', 'payload' keys.
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]

    def search_excluding(
        self,
        query_vector: list[float],
        exclude_ids: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """Search but exclude specific chunk IDs from results.

        Args:
            query_vector: The query embedding vector.
            exclude_ids: Chunk IDs to exclude from results.
            top_k: Number of results to return (before exclusion).

        Returns:
            List of dicts with 'id', 'score', 'payload' keys, excluding
            the specified IDs.
        """
        # Search with a larger top_k to account for exclusions
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k * 3,  # Over-fetch to account for exclusions
            with_payload=True,
            with_vectors=False,
        )

        filtered = [hit for hit in results if hit.payload.get("id") not in exclude_ids]
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in filtered[:top_k]
        ]
