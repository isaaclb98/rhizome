"""Qdrant-backed vector storage."""

from rhizome.vectorstore.client import VectorStoreClient
from rhizome.vectorstore.collection import CollectionManager

__all__ = ["VectorStoreClient", "CollectionManager"]
