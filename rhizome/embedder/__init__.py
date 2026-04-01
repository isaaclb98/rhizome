"""Embedder interface and implementations."""

from rhizome.embedder.base import Embedder
from rhizome.embedder.openai import OpenAIEmbedder
from rhizome.embedder.huggingface import HuggingFaceEmbedder

__all__ = ["Embedder", "OpenAIEmbedder", "HuggingFaceEmbedder"]
