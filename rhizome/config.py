"""Pydantic-validated configuration for Rhizome.

All configuration is driven by environment variables. Config file (config.yaml)
is no longer used directly — env vars take precedence.

Env vars:
    QDRANT_URL           — Qdrant server URL (default: http://localhost:6333)
    QDRANT_COLLECTION    — Qdrant collection name (required)
    QDRANT_API_KEY      — Qdrant API key (optional)
    EMBEDDER_TYPE       — Embedder type: openai, huggingface, or gateway (default: openai)
    OPENAI_API_KEY      — OpenAI API key (required if EMBEDDER_TYPE=openai)
    HF_API_TOKEN        — HuggingFace API token (required if EMBEDDER_TYPE=huggingface)
    HF_MODEL            — HuggingFace model (default: sentence-transformers/all-MiniLM-L6-v2)
    LLM_GATEWAY_URL     — OpenAI-compatible gateway base URL (required if EMBEDDER_TYPE=gateway,
                          and always required for `rhizome idea` / the /idea endpoints)
    EMBEDDING_MODEL     — Embedding model name when using the gateway
    DEFAULT_DEPTH       — Default traversal depth (default: 10)
    EPSILON             — Epsilon-greedy exploration probability (default: 0.1)
    WIKIPEDIA_DEPTH     — PetScan subcategory depth for Wikipedia article discovery (default: 1)
    WIKIPEDIA_CATEGORIES — Wikipedia categories for ingestion, comma-separated (default: Modernism,Postmodernism,Critical theory)
    RHIZOME_CHECKPOINT_PATH — Path to the ingestion checkpoint file (default: .rhizome_checkpoints)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class RhizomeConfig(BaseSettings):
    """Validated configuration for all Rhizome commands.

    All fields map to environment variables. If a field has an alias (ALIAS_NAME),
    the environment variable NAME is used. Fields without a default are required.
    """

    model_config = SettingsConfigDict(
        env_prefix="",          # No prefix — env vars are already uppercase
        populate_by_name=True,   # Allow field name OR alias in code
        case_sensitive=False,
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Vector store
    qdrant_url: str = Field(
        default="http://localhost:6333",
        alias="QDRANT_URL",
    )
    qdrant_collection: str = Field(
        default="modernity-v1",
        alias="QDRANT_COLLECTION",
    )
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")

    # Embedder
    embedder_type: Literal["openai", "huggingface", "gateway"] = Field(
        default="openai",
        alias="EMBEDDER_TYPE",
    )
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    hf_api_token: str | None = Field(default=None, alias="HF_API_TOKEN")
    hf_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="HF_MODEL",
    )

    # Corpus / traversal
    wikipedia_categories: str = Field(
        default="Modernism,Postmodernism,Critical theory",
        alias="WIKIPEDIA_CATEGORIES",
    )
    wikipedia_depth: int = Field(default=1, alias="WIKIPEDIA_DEPTH")
    checkpoint_path: str = Field(
        default=".rhizome_checkpoints",
        alias="RHIZOME_CHECKPOINT_PATH",
    )
    default_depth: int = Field(default=10, alias="DEFAULT_DEPTH")
    epsilon: float = Field(default=0.1, alias="EPSILON")
    top_k: int = Field(default=20, alias="TOP_K")
    temperature: float = Field(default=1.0, alias="TEMPERATURE")
    max_same_article_consecutive: int = Field(default=2, alias="MAX_SAME_ARTICLE_CONSECUTIVE")

    # Idea synthesis (`rhizome idea`) — OpenAI-compatible endpoint for chat and
    # embeddings. No key needed if the endpoint is open.
    llm_gateway_url: str | None = Field(default=None, alias="LLM_GATEWAY_URL")
    llm_gateway_api_key: str | None = Field(default=None, alias="LLM_GATEWAY_API_KEY")
    llm_model: str = Field(
        default="isaac-quality", alias="LLM_MODEL"
    )
    embedding_model: str = Field(
        default="openai/text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    llm_temperature: float = Field(default=0.9, alias="LLM_TEMPERATURE")

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator(
        "qdrant_api_key", "openai_api_key", "hf_api_token", "llm_gateway_api_key", mode="before"
    )
    @classmethod
    def resolve_env_var(cls, v: str | None) -> str | None:
        """Resolve ${VAR} syntax for backward compat with config.yaml env refs.

        If a string value starts with ${ and ends with }, treat it as an
        environment variable reference and resolve it.
        """
        if v is None:
            return None
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_name = v[2:-1]
            return os.environ.get(env_name)
        return v

    @field_validator("wikipedia_categories", "hf_model", mode="before")
    @classmethod
    def parse_comma_separated(cls, v: str | list[str]) -> str:
        """Parse comma-separated string into list of categories; keep as string for Pydantic."""
        if isinstance(v, list):
            return ",".join(v)
        return v

    @field_validator("embedder_type", mode="before")
    @classmethod
    def validate_embedder_type(cls, v: str) -> str:
        """Normalize embedder type to lowercase."""
        if isinstance(v, str):
            v = v.lower().strip()
        if v not in ("openai", "huggingface", "gateway"):
            raise ValueError(
                "EMBEDDER_TYPE must be 'openai', 'huggingface', or 'gateway', "
                f"got '{v}'"
            )
        return v

    # ── Convenience helpers ────────────────────────────────────────────────────

    def require_openai_key(self) -> str:
        """Return OpenAI API key, raising if not set."""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it or switch EMBEDDER_TYPE to 'huggingface'."
            )
        return self.openai_api_key

    def require_hf_token(self) -> str:
        """Return HuggingFace API token, raising if not set."""
        if not self.hf_api_token:
            raise ValueError(
                "HF_API_TOKEN environment variable is not set. "
                "Set it or switch EMBEDDER_TYPE to 'openai'."
            )
        return self.hf_api_token

    def require_qdrant_key(self) -> str | None:
        """Return Qdrant API key if set, else None."""
        return self.qdrant_api_key


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton for CLI commands (avoids re-parsing on every call)
# ─────────────────────────────────────────────────────────────────────────────

_config: RhizomeConfig | None = None


def get_config() -> RhizomeConfig:
    """Return the cached configuration singleton.

    Call this in CLI commands instead of constructing RhizomeConfig each time.
    Raises ValidationError if required env vars are missing.
    """
    global _config
    if _config is None:
        _config = RhizomeConfig()  # BaseSettings reads env vars automatically
    return _config
