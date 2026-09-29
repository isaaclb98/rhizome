"""Qdrant client wrapper for vector search operations."""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from rhizome.corpus.chunker import Chunk


class VectorStoreClient:
    """Wrapper around Qdrant client for vector search operations."""

    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None, collection_name: str = "modernity-v1"):
        self.client = QdrantClient(url=url, api_key=api_key, port=443 if url.startswith("https://") else None, timeout=30)
        self.collection_name = collection_name

    def _query_points(
        self,
        query_vector: list[float],
        limit: int,
        query_filter: Filter | None,
        with_vector: bool,
    ):
        """Run a nearest-neighbour query and return the raw hit objects.

        Uses query_points rather than the removed low-level SearchRequest API.

        Args:
            query_vector: The query embedding vector.
            limit: Maximum number of points to return.
            query_filter: Optional Qdrant filter.
            with_vector: Whether to return stored vectors.

        Returns:
            List of scored point objects.
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=with_vector,
        )
        return list(getattr(response, "points", None) or [])

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        query_filter: Filter | None = None,
        with_vector: bool = True,
    ) -> list[dict]:
        """Search for the nearest chunks to a query vector.

        Args:
            query_vector: The query embedding vector.
            top_k: Number of results to return.
            query_filter: Optional Qdrant filter.
            with_vector: Whether to return stored vectors (default True).

        Returns:
            List of dicts with 'id', 'score', 'payload', and optionally 'vector' keys.
        """
        points = self._query_points(query_vector, top_k, query_filter, with_vector)

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
                "vector": getattr(hit, "vector", None) if with_vector else None,
            }
            for hit in points
        ]

    def search_excluding(
        self,
        query_vector: list[float],
        exclude_ids: list[str],
        top_k: int = 5,
        query_filter: Filter | None = None,
        with_vector: bool = True,
    ) -> list[dict]:
        """Search but exclude specific chunk IDs from results.

        Args:
            query_vector: The query embedding vector.
            exclude_ids: Chunk IDs to exclude from results.
            top_k: Number of results to return (before exclusion).
            query_filter: Optional Qdrant filter to apply alongside exclusion.
            with_vector: Whether to return stored vectors (default True).

        Returns:
            List of dicts with 'id', 'score', 'payload', and optionally 'vector' keys,
            excluding the specified IDs.
        """
        # Over-fetch to account for exclusions
        points = self._query_points(
            query_vector, top_k * 3, query_filter, with_vector
        )

        filtered = [hit for hit in points if (hit.payload or {}).get("id") not in exclude_ids]
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
                "vector": getattr(hit, "vector", None) if with_vector else None,
            }
            for hit in filtered[:top_k]
        ]
