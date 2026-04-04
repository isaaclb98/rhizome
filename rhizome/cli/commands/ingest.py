"""Ingest Wikipedia articles and store chunks in Qdrant."""

import os
import time
import click

from rhizome.config import get_config
from rhizome.corpus.wikipedia_ingester import WikipediaIngester, IngesterError
from rhizome.corpus.chunker import Chunker
from rhizome.embedder.factory import get_embedder
from rhizome.vectorstore.collection import CollectionManager

BATCH_SIZE = 100
LOG_EVERY = 50  # log every N articles ingested


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
    "--categories",
    "categories",
    default=None,
    help="Wikipedia categories to ingest (comma-separated, e.g. --categories Modernism,Postmodernism). "
         "Defaults to WIKIPEDIA_CATEGORIES env var or config.",
)
def ingest(categories: str | None):
    """Ingest Wikipedia articles from PetScan category membership and store chunks in Qdrant.

    Checkpointing is automatic — articles already in the checkpoint file are skipped.
    The checkpoint path is controlled by RHIZOME_CHECKPOINT_PATH (default: .rhizome_checkpoints).

    Example:
        rhizome ingest --categories Modernism,Postmodernism
    """
    cfg = get_config()

    category_list = (
        [c.strip() for c in categories.split(",") if c.strip()]
        if categories
        else cfg.wikipedia_categories
    )
    checkpoint_path = cfg.checkpoint_path

    click.echo(f"[rhizome] Starting ingestion: categories={category_list}, depth={cfg.wikipedia_depth}")
    click.echo(f"[rhizome] Checkpoint: {checkpoint_path}")

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
        click.echo(f"[rhizome] Creating collection: {cfg.qdrant_collection} (vector_size={vector_size})")
        collection_mgr.create_collection(
            collection_name=cfg.qdrant_collection,
            vector_size=vector_size,
            recreate=False,
        )
    else:
        click.echo(f"[rhizome] Using existing collection: {cfg.qdrant_collection}")

    # Load checkpoint — skip articles already ingested from prior runs
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint:
        click.echo(f"[rhizome] Checkpoint: {len(checkpoint)} articles already ingested, will skip")

    # Discover articles via PetScan
    ingester = WikipediaIngester(
        categories=category_list,
        depth=cfg.wikipedia_depth,
        chunker=Chunker(),
    )
    start_discovery = time.monotonic()
    discovered_titles = ingester._discover_articles()
    click.echo(f"[rhizome] PetScan: {len(discovered_titles)} articles found ({time.monotonic() - start_discovery:.1f}s)")

    to_ingest = len(discovered_titles) - len(checkpoint & set(discovered_titles))
    click.echo(f"[rhizome] Ingesting: {to_ingest} new articles (+ {len(checkpoint & set(discovered_titles))} already checkpointed)")

    total_chunks = 0
    total_articles = 0
    batch_chunks = []
    batch_texts = []
    start_ingest = time.monotonic()

    try:
        with click.progressbar(
            length=to_ingest,
            label="Ingesting",
            show_eta=True,
            show_percent=True,
        ) as bar:
            for chunk in ingester.ingest():
                # Skip articles already ingested (checkpoint is the source of truth)
                if chunk.article_title in checkpoint:
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
                            bar.update(1)

                            # Periodic verbose log every LOG_EVERY articles
                            if total_articles % LOG_EVERY == 0:
                                bar.stop()
                                elapsed = time.monotonic() - start_ingest
                                rate = total_articles / elapsed if elapsed > 0 else 0
                                click.echo(
                                    f"[rhizome] {total_articles}/{to_ingest} articles "
                                    f"({total_chunks} chunks, {rate:.1f} articles/s)",
                                    err=True,
                                )
                                bar.render_progress()

                    batch_chunks = []
                    batch_texts = []

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
        click.echo(f"[rhizome] Error during ingestion: {e}", err=True)
        raise click.Abort()

    elapsed = time.monotonic() - start_ingest
    click.echo(
        f"[rhizome] Done: {total_articles} articles, {total_chunks} chunks "
        f"in {elapsed:.1f}s ({total_articles/elapsed:.1f} articles/s)"
    )
