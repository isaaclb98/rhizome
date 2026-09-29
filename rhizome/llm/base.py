"""LLM client interface.

The idea agent talks to models only through LLMClient, so the transport can be
swapped (gateway, OpenAI, Anthropic) without touching the agent loop.
"""

from abc import abstractmethod
from typing import Any, Protocol


class LLMError(Exception):
    """Raised when a model call fails or returns unusable output."""


class TruncatedResponseError(LLMError):
    """Raised when the model hit its token budget before finishing."""


class LLMClient(Protocol):
    """Interface for chat-completion providers."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Return the assistant's text reply.

        Args:
            messages: Chat messages as {"role": ..., "content": ...}.
            temperature: Sampling temperature.
            max_tokens: Output token budget.

        Returns:
            Assistant message content.

        Raises:
            LLMError: On transport failure.
        """
        ...

    @abstractmethod
    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        retries: int = 1,
    ) -> Any:
        """Return a parsed JSON object from the model.

        Implementations must tolerate fenced output and surrounding prose, and
        should re-prompt once with the parse error when the first reply is not
        valid JSON.

        Args:
            messages: Chat messages.
            temperature: Sampling temperature.
            max_tokens: Output token budget.
            retries: Number of re-prompts on parse failure.

        Returns:
            Parsed JSON (dict or list).

        Raises:
            LLMError: If the model never returns parseable JSON.
        """
        ...
