"""Embedder factory for creating the appropriate embedder based on configuration."""

from rhizome.embedder import Embedder, OpenAIEmbedder, HuggingFaceEmbedder, EmbeddingError


def get_embedder(
    embedder_type: str,
    openai_api_key: str | None = None,
    hf_api_token: str | None = None,
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    gateway_url: str | None = None,
    gateway_model: str = "openai/text-embedding-3-small",
    gateway_api_key: str | None = None,
) -> Embedder:
    """Create an embedder based on the embedder type.

    Args:
        embedder_type: One of "openai", "huggingface", or "gateway".
        openai_api_key: OpenAI API key (required if embedder_type is "openai").
        hf_api_token: HuggingFace API token (required if embedder_type is "huggingface").
        hf_model: HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2).
        gateway_url: Base URL of an OpenAI-compatible gateway
            (required if embedder_type is "gateway").
        gateway_model: Embedding model exposed by the gateway.
        gateway_api_key: Optional bearer token; omit when the gateway is open.

    Returns:
        An Embedder instance.

    Raises:
        EmbeddingError: If embedder_type is invalid or required credentials are missing.
    """
    et = embedder_type.lower().strip()

    if et == "openai":
        if not openai_api_key:
            raise EmbeddingError(
                "OpenAI API key is required when EMBEDDER_TYPE=openai. "
                "Set the OPENAI_API_KEY environment variable."
            )
        return OpenAIEmbedder(api_key=openai_api_key)

    if et == "huggingface":
        if not hf_api_token:
            raise EmbeddingError(
                "HuggingFace API token is required when EMBEDDER_TYPE=huggingface. "
                "Set the HF_API_TOKEN environment variable."
            )
        return HuggingFaceEmbedder(api_token=hf_api_token, model=hf_model)

    if et == "gateway":
        if not gateway_url:
            raise EmbeddingError(
                "LLM_GATEWAY_URL is required when EMBEDDER_TYPE=gateway."
            )
        from rhizome.gateway import GatewayEmbedder

        return GatewayEmbedder(
            base_url=gateway_url,
            model=gateway_model,
            api_key=gateway_api_key,
        )

    raise EmbeddingError(
        f"EMBEDDER_TYPE must be 'openai', 'huggingface', or 'gateway', got '{embedder_type}'"
    )
