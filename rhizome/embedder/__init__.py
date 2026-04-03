"""Embedder interface and implementations."""

from rhizome.embedder.base import Embedder, EmbeddingError
from rhizome.embedder.openai import OpenAIEmbedder
from rhizome.embedder.huggingface import HuggingFaceEmbedder

__all__ = ["Embedder", "EmbeddingError", "OpenAIEmbedder", "HuggingFaceEmbedder"]
