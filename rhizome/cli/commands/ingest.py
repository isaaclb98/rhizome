"""Ingest Wikipedia articles and store chunks in Qdrant."""

import os
import yaml
import click

from rhizome.corpus.wikipedia_ingester import WikipediaIngester, IngesterError
from rhizome.corpus.chunker import Chunker
from rhizome.embedder.huggingface import HuggingFaceEmbedder
from rhizome.vectorstore.collection import CollectionManager


@click.command()
@click.option(
    "--config",
    default="config.yaml",
    type=click.Path(exists=True),
    help="Path to config.yaml",
)
@click.option(
    "--domain",
    "domains",
    multiple=True,
    help="Wikipedia domain(s) to ingest. Can be specified multiple times (overrides config)",
)
@click.option(
    "--max-articles",
    type=int,
    help="Maximum articles to ingest (overrides config)",
)
def ingest(config: str, domains: tuple[str, ...] | None, max_articles: int | None):
    """Ingest Wikipedia articles and store chunks in Qdrant.

    Reads configuration from config.yaml. Run this before `rhizome traverse`.

    Example:
        rhizome ingest --domain Modernism --domain Postmodernism --max-articles 500
    """
    # Load config
    with open(config) as f:
        cfg = yaml.safe_load(f)

    traversal_cfg = cfg.get("traversal", {})
    corpus_cfg = cfg.get("corpus", {})
    hf_cfg = cfg.get("huggingface", {})

    domain_cfg = corpus_cfg.get("domain", "Modernism")
    domain_list = list(domains) if domains else ([domain_cfg] if isinstance(domain_cfg, list) else [domain_cfg])
    max_articles = max_articles or corpus_cfg.get("max_articles", 500)
    collection_name = traversal_cfg.get("collection", "modernity-v1")

    # Resolve API token
    api_token = os.environ.get("HF_API_TOKEN") or hf_cfg.get("api_token")
    if api_token and api_token.startswith("${"):
        # Environment variable reference — try to resolve it
        env_var = api_token[2:-1]  # strip ${ and }
        api_token = os.environ.get(env_var)

    click.echo(f"Starting ingestion: domains={domain_list}, max_articles={max_articles}")

    # Set up components
    embedder = HuggingFaceEmbedder(api_token=api_token)
    collection_mgr = CollectionManager()

    # Ensure collection exists
    if not collection_mgr.collection_exists(collection_name):
        click.echo(f"Creating collection: {collection_name}")
        collection_mgr.create_collection(
            collection_name=collection_name,
            vector_size=embedder.vector_size(),
            recreate=False,
        )
    else:
        click.echo(f"Using existing collection: {collection_name}")

    # Ingest articles
    ingester = WikipediaIngester(
        domains=domain_list,
        max_articles=max_articles,
        chunker=Chunker(),
    )

    total_chunks = 0
    total_articles = 0
    batch_chunks = []
    batch_vectors = []

    try:
        for chunk in ingester.ingest():
            # Embed the chunk text
            embedding = embedder.embed([chunk.text])[0]
            batch_chunks.append(chunk)
            batch_vectors.append(embedding)

            # Upsert in batches to avoid overwhelming Qdrant
            if len(batch_chunks) >= 50:
                collection_mgr.upsert_chunks(collection_name, batch_chunks, batch_vectors)
                total_chunks += len(batch_chunks)
                total_articles += 1  # approximate
                click.echo(f"  Upserted {len(batch_chunks)} chunks (total: {total_chunks})")
                batch_chunks = []
                batch_vectors = []

        # Final batch
        if batch_chunks:
            collection_mgr.upsert_chunks(collection_name, batch_chunks, batch_vectors)
            total_chunks += len(batch_chunks)
            click.echo(f"  Upserted {len(batch_chunks)} chunks (total: {total_chunks})")

    except IngesterError as e:
        click.echo(f"Error during ingestion: {e}", err=True)
        raise click.Abort()

    click.echo(f"Ingestion complete: {total_chunks} chunks from ~{total_articles} articles")
