"""LLM access for the idea agent."""

from rhizome.llm.base import LLMClient, LLMError
from rhizome.llm.gateway import GatewayLLM, GatewayEmbedder

__all__ = ["LLMClient", "LLMError", "GatewayLLM", "GatewayEmbedder"]
