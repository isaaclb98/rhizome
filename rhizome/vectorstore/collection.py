"""Qdrant collection management: create, upsert, delete."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from rhizome.corpus.chunker import Chunk


class CollectionManager:
    """Manages Qdrant collections for the rhizome vector store."""

    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None):
        # When using HTTPS, Qdrant Cloud serves on port 443 (not the default 6333)
        port = 443 if url.startswith("https://") else None
        self.client = QdrantClient(url=url, api_key=api_key, port=port)

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance: Distance = Distance.COSINE,
        recreate: bool = False,
    ):
        """Create a Qdrant collection.

        Args:
            collection_name: Name of the collection.
            vector_size: Dimension of the embedding vectors.
            distance: Distance metric (COSINE, EUCLID, DOT).
            recreate: If True, delete existing collection first.
        """
        if recreate:
            self.client.delete_collection(collection_name=collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ):
        """Insert or update chunks with their embedding vectors.

        Args:
            collection_name: Name of the collection.
            chunks: List of Chunk objects.
            vectors: List of embedding vectors (same order as chunks).

        Raises:
            ValueError: If number of chunks doesn't match number of vectors.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunk count ({len(chunks)}) != vector count ({len(vectors)})")

        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "id": chunk.id,
                    "text": chunk.text,
                    "article_title": chunk.article_title,
                    "article_url": chunk.article_url,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return self.client.collection_exists(collection_name=collection_name)

    def delete_chunks_by_article(self, collection_name: str, article_title: str):
        """Delete all chunks for an article using delete-by-filter.

        Args:
            collection_name: Name of the collection.
            article_title: Article title to delete chunks for.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="article_title", match=MatchValue(value=article_title))]
            ),
        )
