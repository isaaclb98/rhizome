"""Traverse Wikipedia embeddings and produce a narrative document."""

import click

from rhizome.config import get_config
from rhizome.embedder.factory import get_embedder
from rhizome.embedder import EmbeddingError
from rhizome.vectorstore.client import VectorStoreClient
from rhizome.vectorstore.collection import CollectionManager
from rhizome.traversal.engine import TraversalEngine, TraversalError as TraversalErr
from rhizome.traversal.config import TraversalConfig
from rhizome.stitching.formatter import stitch_to_markdown


@click.command()
@click.argument("concept")
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
    "--temperature",
    type=float,
    help="Softmax temperature for exploit path: 0=greedy, 1=natural, 2+=flat (overrides config)",
)
@click.option(
    "--max-same-article-consecutive",
    type=int,
    help="Hard block: max consecutive chunks from the same article (0=disabled, overrides config)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file (default: stdout)",
)
def traverse(
    concept: str,
    depth: int | None,
    epsilon: float | None,
    top_k: int | None,
    temperature: float | None,
    max_same_article_consecutive: int | None,
    output: str | None,
):
    """Traverse Wikipedia embeddings and produce a narrative document.

    Takes a starting concept (keyword or phrase) and walks through the
    vector space rhizomatically, producing a markdown document with
    citations.

    Example:
        rhizome traverse "the tension between modernism and postmodernism" --depth 8
    """
    cfg = get_config()

    depth = depth if depth is not None else cfg.default_depth
    epsilon = epsilon if epsilon is not None else cfg.epsilon
    top_k = top_k if top_k is not None else cfg.top_k
    temperature = temperature if temperature is not None else cfg.temperature
    max_same_article_consecutive = (
        max_same_article_consecutive
        if max_same_article_consecutive is not None
        else cfg.max_same_article_consecutive
    )

    click.echo(f"Traversing: concept='{concept}', depth={depth}, epsilon={epsilon}, temperature={temperature}")

    # Set up components
    embedder = get_embedder(
        embedder_type=cfg.embedder_type,
        openai_api_key=cfg.openai_api_key,
        hf_api_token=cfg.hf_api_token,
        hf_model=cfg.hf_model,
    )
    vector_store = VectorStoreClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection_name=cfg.qdrant_collection,
    )
    collection_mgr = CollectionManager(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)

    # Validate embedder vector size matches collection
    # Note: actual validation requires fetching collection info from Qdrant.
    # The embedder's vector_size() is used at ingest time to create the collection
    # with the correct dimension, so a mismatch at traverse time indicates a config
    # error rather than a runtime problem.

    # Check collection exists
    if not collection_mgr.collection_exists(cfg.qdrant_collection):
        click.echo(
            f"Collection '{cfg.qdrant_collection}' not found. Run `rhizome ingest` first.",
            err=True,
        )
        raise click.Abort()

    # Configure and run traversal
    config = TraversalConfig(
        depth=depth,
        epsilon=epsilon,
        top_k=top_k,
        collection_name=cfg.qdrant_collection,
        temperature=temperature,
        max_same_article_consecutive=max_same_article_consecutive,
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
