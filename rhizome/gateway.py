"""OpenAI-compatible gateway transport: embeddings and chat completions.

The corpus must be queried with the same embedding model that built it, so the
embedder here is plumbing rather than a provider choice — set EMBEDDING_MODEL
to whatever the collection was ingested with (1536-dim for text-embedding-3-small).

Reasoning models need two accommodations, both learned from real runs:

- they prepend a thinking block that is not the payload and must be stripped
  before the answer is read
- they can spend an entire token budget reasoning and return nothing, which
  surfaces as finish_reason == "length" and must be retried with more room
  rather than treated as an empty reply

Structured output (``response_format={"type": "json_object"}``) is supported
on the gateways we use; the runtime coerces Claude via its tool-use path and
OpenAI via JSON mode. The gateway itself stays format-agnostic — callers opt
in by passing ``response_format`` and parse the result.
"""

from __future__ import annotations

import json
import logging
import re

import requests

from rhizome.embedder.base import Embedder, EmbeddingError

logger = logging.getLogger(__name__)

MAX_TOKEN_BUDGET = 32768


class GatewayError(Exception):
    """Raised when a gateway call fails or returns an unusable reply."""


def strip_thinking(text: str | None) -> str:
    """Remove reasoning blocks a thinking model emits before its answer.

    Handles unterminated blocks too: a model cut off mid-reasoning leaves an
    open tag, and keeping the trailing fragment is better than discarding the
    reply outright.

    Args:
        text: Raw model output, or None when the gateway omitted content.

    Returns:
        Text with reasoning blocks removed; "" for None or empty input.
    """
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if cleaned == text:
        cleaned = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return cleaned


# JSON-mode reply schema for synthesis outputs. Kept here so the parser and
# the prompt (which mirrors this shape to the model) cannot drift apart.
THESIS_JSON_FIELDS = ("main_thesis", "content")


def parse_thesis_json(text: str | None) -> dict:
    """Parse a JSON-mode synthesis reply into the canonical thesis dict.

    The model is asked for ``{"main_thesis": str, "content": str}`` and is
    expected to honor that schema in response_format=json_object mode. In
    practice it sometimes slips: it wraps the JSON in a code fence, leaves
    trailing prose after the closing brace, omits a field, or hands back the
    schema but with ``content`` empty. This helper tolerates those shapes
    without raising, so a sloppy reply still produces a run.

    Args:
        text: Raw reply from the gateway, with reasoning blocks already
            stripped.

    Returns:
        A dict with both ``main_thesis`` and ``content`` keys. Empty string
        for any field the model omitted. If the reply cannot be located as
        JSON at all, the entire stripped text is returned as ``content`` and
        ``main_thesis`` is empty — better than losing the run.
    """
    out: dict = {"main_thesis": "", "content": ""}
    if not text:
        return out

    candidate = text.strip()

    # Strip a single wrapping ```json ... ``` fence if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()

    # Find the outermost JSON object even if there is trailing prose.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start:end + 1]

    parsed: object = None
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        for key in THESIS_JSON_FIELDS:
            value = parsed.get(key)
            if isinstance(value, str):
                out[key] = value
    else:
        # Reply was not JSON. Treat the whole thing as content so the run
        # still produces something; main_thesis stays empty.
        out["content"] = text.strip()

    return out


class GatewayLLM:
    """Chat completions against an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str = "agy/claude-opus-4-6-thinking-high",
        api_key: str | None = None,
        timeout: int = 600,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, payload: dict) -> dict:
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise GatewayError(f"gateway request failed: {exc}") from exc

        if response.status_code != 200:
            raise GatewayError(
                f"gateway error {response.status_code}: {response.text[:500]}"
            )
        return response.json()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.9,
        max_tokens: int = 8192,
        retries: int = 2,
        response_format: dict | None = None,
    ) -> str:
        """Return the assistant's reply, retrying with more room if truncated.

        Args:
            messages: Chat messages as {"role": ..., "content": ...}.
            temperature: Sampling temperature.
            max_tokens: Initial output token budget.
            retries: Retries allowed when the reply is truncated.
            response_format: Optional ``{"type": "json_object"}`` (or similar)
                passed straight through to the gateway. When set, the reply
                comes back as a JSON string the caller parses; this method
                does not enforce schema or parse for you.

        Returns:
            Assistant reply text with reasoning blocks stripped.

        Raises:
            GatewayError: On transport failure or a reply that never completes.
        """
        budget = max_tokens
        for _attempt in range(retries + 1):
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": budget,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            data = self._post(payload)
            try:
                choice = data["choices"][0]
                content = strip_thinking(choice["message"]["content"])
                finish_reason = choice.get("finish_reason")
            except (KeyError, IndexError, TypeError) as exc:
                raise GatewayError(f"unexpected gateway response shape: {exc}") from exc

            if content.strip() and finish_reason != "length":
                return content

            if finish_reason != "length":
                raise GatewayError("gateway returned an empty completion")

            if budget >= MAX_TOKEN_BUDGET:
                raise GatewayError(
                    f"reply still truncated at the {budget}-token ceiling"
                )
            budget = min(budget * 2, MAX_TOKEN_BUDGET)
            logger.warning(
                "reply truncated; retrying with max_tokens=%d", budget
            )

        raise GatewayError("reply never completed")


class GatewayEmbedder(Embedder):
    """Embeddings against an OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str = "openai/text-embedding-3-small",
        api_key: str | None = None,
        timeout: int = 120,
        batch_size: int = 128,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, batching to keep request payloads bounded.

        Args:
            texts: Strings to embed.

        Returns:
            Embeddings in the same order as the input texts.

        Raises:
            EmbeddingError: On transport failure or a count mismatch.
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"model": self.model, "input": batch},
                    headers=self._headers(),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise EmbeddingError(f"gateway embedding request failed: {exc}") from exc

            if response.status_code != 200:
                raise EmbeddingError(
                    f"gateway embedding error {response.status_code}: {response.text[:500]}"
                )

            try:
                data = response.json()["data"]
            except (ValueError, KeyError, TypeError) as exc:
                raise EmbeddingError(f"unexpected embeddings response: {exc}") from exc

            ordered = sorted(data, key=lambda item: item.get("index", 0))
            if len(ordered) != len(batch):
                raise EmbeddingError(
                    f"expected {len(batch)} embeddings, got {len(ordered)}"
                )
            vectors.extend(item["embedding"] for item in ordered)

        return vectors
