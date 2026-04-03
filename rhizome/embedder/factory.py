"""Embedder factory for creating the appropriate embedder based on configuration."""

from rhizome.embedder import Embedder, OpenAIEmbedder, HuggingFaceEmbedder, EmbeddingError


def get_embedder(
    embedder_type: str,
    openai_api_key: str | None = None,
    hf_api_token: str | None = None,
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Embedder:
    """Create an embedder based on the embedder type.

    Args:
        embedder_type: Either "openai" or "huggingface".
        openai_api_key: OpenAI API key (required if embedder_type is "openai").
        hf_api_token: HuggingFace API token (required if embedder_type is "huggingface").
        hf_model: HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2).

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

    raise EmbeddingError(
        f"EMBEDDER_TYPE must be 'openai' or 'huggingface', got '{embedder_type}'"
    )
