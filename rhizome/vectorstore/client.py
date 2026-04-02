"""Qdrant client wrapper for vector search operations."""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from rhizome.corpus.chunker import Chunk


class VectorStoreClient:
    """Wrapper around Qdrant client for vector search operations."""

    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None, collection_name: str = "modernity-v1"):
        self.client = QdrantClient(url=url, api_key=api_key, port=443 if url.startswith("https://") else None)
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
        from qdrant_client.http import models

        search_req = models.SearchRequest(
            vector=query_vector,
            limit=top_k,
            filter=query_filter,
            with_payload=True,
            with_vector=False,
        )
        response = self.client.http.search_api.search_points(
            collection_name=self.collection_name,
            search_request=search_req,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in (response.result or [])
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
        from qdrant_client.http import models

        # Over-fetch to account for exclusions
        search_req = models.SearchRequest(
            vector=query_vector,
            limit=top_k * 3,
            with_payload=True,
            with_vector=False,
        )
        response = self.client.http.search_api.search_points(
            collection_name=self.collection_name,
            search_request=search_req,
        )

        hits = response.result or []
        filtered = [hit for hit in hits if (hit.payload or {}).get("id") not in exclude_ids]
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in filtered[:top_k]
        ]
