"""Ingest Wikipedia articles and store chunks in Qdrant."""

import os
import click

from rhizome.config import get_config
from rhizome.corpus.wikipedia_ingester import WikipediaIngester, IngesterError
from rhizome.corpus.chunker import Chunker
from rhizome.embedder.factory import get_embedder
from rhizome.vectorstore.collection import CollectionManager

BATCH_SIZE = 100


def _load_checkpoint(path: str) -> set[str]:
    """Load set of already-ingested article titles from checkpoint file."""
    if not os.path.exists(path):
        return set()
    return set(line.strip() for line in open(path) if line.strip())


def _append_checkpoint(path: str, title: str) -> None:
    """Append a successfully-ingested article title to the checkpoint file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(title + "\n")


@click.command()
@click.option(
    "--domain",
    "domains",
    multiple=True,
    help="Wikipedia domain(s) to ingest. Can be specified multiple times (overrides config)",
)
def ingest(domains: tuple[str, ...] | None):
    """Ingest Wikipedia articles and store chunks in Qdrant.

    Checkpointing is automatic — articles already in the checkpoint file are skipped.
    The checkpoint path is controlled by RHIZOME_CHECKPOINT_PATH (default: .rhizome_checkpoints).

    Run this before `rhizome traverse`.

    Example:
        rhizome ingest --domain Modernism --domain Postmodernism
    """
    cfg = get_config()

    domain_list = list(domains) if domains else cfg.wikipedia_domains
    checkpoint_path = cfg.checkpoint_path

    click.echo(f"Starting ingestion: domains={domain_list}, depth={cfg.wikipedia_depth}")
    click.echo(f"Checkpoint: {checkpoint_path}")

    # Set up components
    embedder = get_embedder(
        embedder_type=cfg.embedder_type,
        openai_api_key=cfg.openai_api_key,
        hf_api_token=cfg.hf_api_token,
        hf_model=cfg.hf_model,
    )
    collection_mgr = CollectionManager(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)

    # Ensure collection exists
    vector_size = embedder.vector_size()
    if not collection_mgr.collection_exists(cfg.qdrant_collection):
        click.echo(f"Creating collection: {cfg.qdrant_collection} (vector_size={vector_size})")
        collection_mgr.create_collection(
            collection_name=cfg.qdrant_collection,
            vector_size=vector_size,
            recreate=False,
        )
    else:
        click.echo(f"Using existing collection: {cfg.qdrant_collection}")

    # Load checkpoint — skip articles already ingested from prior runs
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint:
        click.echo(f"Loaded checkpoint: {len(checkpoint)} articles already ingested, will skip")

    # Discover articles first (to get actual count for progress bar)
    ingester = WikipediaIngester(
        domains=domain_list,
        depth=cfg.wikipedia_depth,
        chunker=Chunker(),
    )
    discovered_titles = ingester._discover_articles()
    article_count = len(discovered_titles)

    total_chunks = 0
    total_articles = 0
    batch_chunks = []
    batch_texts = []

    try:
        with click.progressbar(
            length=article_count,
            label="Ingesting articles",
            show_eta=True,
            show_percent=True,
        ) as bar:
            for chunk in ingester.ingest():
                # Skip articles already ingested (checkpoint is the source of truth)
                if chunk.article_title in checkpoint:
                    bar.update(1)
                    continue

                batch_chunks.append(chunk)
                batch_texts.append(chunk.text)

                # Upsert in batches to avoid overwhelming Qdrant and to batch embeddings
                if len(batch_chunks) >= BATCH_SIZE:
                    vectors = embedder.embed(batch_texts)
                    collection_mgr.upsert_chunks(cfg.qdrant_collection, batch_chunks, vectors)
                    total_chunks += len(batch_chunks)
                    # Record each new article in checkpoint after successful upsert
                    for c in batch_chunks:
                        if c.article_title not in checkpoint:
                            _append_checkpoint(checkpoint_path, c.article_title)
                            checkpoint.add(c.article_title)
                            total_articles += 1
                    batch_chunks = []
                    batch_texts = []
                    bar.update(1)

            # Final batch
            if batch_chunks:
                vectors = embedder.embed(batch_texts)
                collection_mgr.upsert_chunks(cfg.qdrant_collection, batch_chunks, vectors)
                total_chunks += len(batch_chunks)
                for c in batch_chunks:
                    if c.article_title not in checkpoint:
                        _append_checkpoint(checkpoint_path, c.article_title)
                        checkpoint.add(c.article_title)
                        total_articles += 1
                bar.update(1)

    except IngesterError as e:
        click.echo(f"Error during ingestion: {e}", err=True)
        raise click.Abort()

    click.echo(f"Ingestion complete: {total_chunks} chunks from {total_articles} articles")
