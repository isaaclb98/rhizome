"""Traverse Wikipedia embeddings and produce a narrative document."""

import os
import yaml
import click

from rhizome.embedder.openai import OpenAIEmbedder, EmbeddingError
from rhizome.vectorstore.client import VectorStoreClient
from rhizome.vectorstore.collection import CollectionManager
from rhizome.traversal.engine import TraversalEngine, TraversalError as TraversalErr
from rhizome.traversal.config import TraversalConfig
from rhizome.stitching.formatter import stitch_to_markdown


@click.command()
@click.argument("concept")
@click.option(
    "--config",
    default="config.yaml",
    type=click.Path(exists=True),
    help="Path to config.yaml",
)
@click.option(
    "--depth",
    type=int,
    help="Maximum traversal depth (overrides config)",
)
@click.option(
    "--epsilon",
    type=float,
    help="Exploration probability 0.0-1.0 (overrides config)",
)
@click.option(
    "--top-k",
    type=int,
    help="Number of candidates per step (overrides config)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file (default: stdout)",
)
def traverse(
    concept: str,
    config: str,
    depth: int | None,
    epsilon: float | None,
    top_k: int | None,
    output: str | None,
):
    """Traverse Wikipedia embeddings and produce a narrative document.

    Takes a starting concept (keyword or phrase) and walks through the
    vector space rhizomatically, producing a markdown document with
    citations.

    Example:
        rhizome traverse "the tension between modernism and postmodernism" --depth 8
    """
    # Load config
    with open(config) as f:
        cfg = yaml.safe_load(f)

    traversal_cfg = cfg.get("traversal", {})
    vectorstore_cfg = cfg.get("vectorstore", {})
    openai_cfg = cfg.get("openai", {})

    depth = depth if depth is not None else traversal_cfg.get("depth", 8)
    epsilon = epsilon if epsilon is not None else traversal_cfg.get("epsilon", 0.1)
    top_k = top_k if top_k is not None else traversal_cfg.get("top_k", 5)
    collection_name = vectorstore_cfg.get("collection", "modernity-v1")

    # Resolve Qdrant URL and API key
    qdrant_url = vectorstore_cfg.get("url", "http://localhost:6333")
    qdrant_api_key = vectorstore_cfg.get("api_key")
    if qdrant_api_key is not None and qdrant_api_key.startswith("${"):
        env_var = qdrant_api_key[2:-1]
        qdrant_api_key = os.environ.get(env_var)

    # Resolve OpenAI API token
    api_token = os.environ.get("OPENAI_API_KEY")
    if api_token is None:
        api_token = openai_cfg.get("api_key")
    if api_token is not None and api_token.startswith("${"):
        env_var = api_token[2:-1]
        api_token = os.environ.get(env_var)

    click.echo(f"Traversing: concept='{concept}', depth={depth}, epsilon={epsilon}")

    # Set up components
    embedder = OpenAIEmbedder(api_key=api_token, model=openai_cfg.get("model", "text-embedding-3-small"))
    vector_store = VectorStoreClient(url=qdrant_url, api_key=qdrant_api_key, collection_name=collection_name)
    collection_mgr = CollectionManager(url=qdrant_url, api_key=qdrant_api_key)

    # Validate embedder vector size matches collection
    expected_size = vectorstore_cfg.get("vector_size", 1536)
    actual_size = embedder.vector_size()
    if actual_size != expected_size:
        click.echo(
            f"Vector dimension mismatch: embedder produces {actual_size}-dim vectors "
            f"but collection '{collection_name}' expects {expected_size}-dim. "
            f"Check your embedding model and vector_size in config.yaml.",
            err=True,
        )
        raise click.Abort()

    # Check collection exists
    if not collection_mgr.collection_exists(collection_name):
        click.echo(
            f"Collection '{collection_name}' not found. Run `rhizome ingest` first.",
            err=True,
        )
        raise click.Abort()

    # Configure and run traversal
    config = TraversalConfig(
        depth=depth,
        epsilon=epsilon,
        top_k=top_k,
        collection_name=collection_name,
    )
    engine = TraversalEngine(embedder=embedder, vector_store=vector_store, config=config)

    try:
        path = engine.traverse(concept)
    except TraversalErr as e:
        click.echo(f"Traversal error: {e}", err=True)
        raise click.Abort()
    except EmbeddingError as e:
        click.echo(f"Embedding error: {e}", err=True)
        raise click.Abort()

    if not path:
        click.echo("No path generated. The corpus may be too small or the concept too specific.")
        raise click.Abort()

    # Produce markdown
    markdown = stitch_to_markdown(concept, path)

    if output:
        with open(output, "w") as f:
            f.write(markdown)
        click.echo(f"Output written to: {output}")
    else:
        click.echo(markdown)
