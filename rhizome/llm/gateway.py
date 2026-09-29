"""OpenAI-compatible gateway transport for LLM calls and embeddings."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

from rhizome.embedder.base import Embedder, EmbeddingError
from rhizome.llm.base import LLMClient, LLMError, TruncatedResponseError

logger = logging.getLogger(__name__)


def strip_thinking(text: str | None) -> str:
    """Remove reasoning blocks a thinking model emits before its answer.

    Reasoning models prepend a thinking block that is not part of the payload.
    It must be discarded before JSON extraction, or the parser sees prose and
    fails. An unterminated block — the model was cut off mid-reasoning — is
    also removed so any trailing content stays parseable.

    Args:
        text: Raw model output, or None when the gateway omitted content.

    Returns:
        Text with every reasoning block removed; "" for None or empty input.
    """
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if cleaned == text:
        cleaned = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return cleaned


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply that may carry fences or surrounding prose.

    Handles bare JSON, ```json fences, and prose-prefixed payloads by scanning
    for the first balanced top-level object or array.

    Args:
        text: Raw model output.

    Returns:
        Parsed JSON value.

    Raises:
        LLMError: If no balanced JSON structure can be parsed.
    """
    if not text or not text.strip():
        raise LLMError("model returned an empty response")

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidates: list[str] = []
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text.strip())
    candidates.extend(_balanced_spans(text))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise LLMError(f"could not parse JSON from model output: {text[:400]!r}")


def _balanced_spans(text: str) -> list[str]:
    """Return balanced {...} and [...] spans found in text, outermost first."""
    spans: list[str] = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    spans.append(text[start : i + 1])
                    break
    return spans


class GatewayLLM(LLMClient):
    """Chat-completion client for an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str = "auto/best-reasoning",
        api_key: str | None = None,
        timeout: int = 300,
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

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"gateway request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(
                f"gateway error {response.status_code}: {response.text[:500]}"
            )

        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected gateway response shape: {exc}") from exc

        content = strip_thinking(content or "")
        if not content.strip():
            if finish_reason == "length":
                raise TruncatedResponseError(
                    f"model spent its entire {max_tokens}-token budget on reasoning "
                    "and produced no answer"
                )
            raise LLMError("gateway returned an empty completion")

        if finish_reason == "length":
            raise TruncatedResponseError(
                f"model output was cut off at {max_tokens} tokens; the reply is incomplete"
            )
        return content

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        retries: int = 1,
    ) -> Any:
        msgs = list(messages)
        last_error: Exception | None = None
        budget = max_tokens

        for _attempt in range(retries + 1):
            try:
                raw = self.complete(msgs, temperature=temperature, max_tokens=budget)
            except TruncatedResponseError as exc:
                last_error = exc
                logger.warning("truncated at %d tokens, raising budget: %s", budget, exc)
                budget = min(budget * 2, 32768)
                msgs = list(messages)
                continue

            try:
                return extract_json(raw)
            except LLMError as exc:
                last_error = exc
                msgs = list(messages) + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON. "
                            f"Parse error: {exc}. "
                            "Reply again with ONLY the JSON object — no reasoning, "
                            "no thinking blocks, no fences, no commentary."
                        ),
                    },
                ]

        raise LLMError(f"model never returned valid JSON: {last_error}")


class GatewayEmbedder(Embedder):
    """Embedder for an OpenAI-compatible /v1/embeddings endpoint."""

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
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = {"model": self.model, "input": batch}

            try:
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json=payload,
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
                raise EmbeddingError(f"unexpected embeddings response shape: {exc}") from exc

            ordered = sorted(data, key=lambda item: item.get("index", 0))
            batch_vectors = [item["embedding"] for item in ordered]
            if len(batch_vectors) != len(batch):
                raise EmbeddingError(
                    f"expected {len(batch)} embeddings, got {len(batch_vectors)}"
                )
            vectors.extend(batch_vectors)

        return vectors
