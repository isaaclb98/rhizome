"""Ingest Wikipedia articles and store chunks in Qdrant."""

import click

from rhizome.config import get_config
from rhizome.corpus.wikipedia_ingester import WikipediaIngester, IngesterError
from rhizome.corpus.chunker import Chunker
from rhizome.embedder.factory import get_embedder
from rhizome.vectorstore.collection import CollectionManager

BATCH_SIZE = 100


@click.command()
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
def ingest(domains: tuple[str, ...] | None, max_articles: int | None):
    """Ingest Wikipedia articles and store chunks in Qdrant.

    Reads configuration from environment variables. Run this before `rhizome traverse`.

    Example:
        rhizome ingest --domain Modernism --domain Postmodernism --max-articles 500
    """
    cfg = get_config()

    domain_list = list(domains) if domains else cfg.wikipedia_domains
    max_articles = max_articles or 500

    click.echo(f"Starting ingestion: domains={domain_list}, max_articles={max_articles}")

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

    # Discover articles first (to get actual count for progress bar)
    ingester = WikipediaIngester(
        domains=domain_list,
        max_articles=max_articles,
        depth=1,
        chunker=Chunker(),
    )
    discovered_titles = ingester._discover_articles()
    actual_article_count = min(len(discovered_titles), max_articles)

    total_chunks = 0
    total_articles = 0
    seen_titles: set[str] = set()
    batch_chunks = []
    batch_texts = []

    try:
        with click.progressbar(
            length=actual_article_count,
            label="Ingesting articles",
            show_eta=True,
            show_percent=True,
        ) as bar:
            for chunk in ingester.ingest():
                # Track new articles — delete any stale chunks from prior ingest
                if chunk.article_title not in seen_titles:
                    total_articles += 1
                    seen_titles.add(chunk.article_title)
                    # Remove old chunks for this article to prevent orphans
                    collection_mgr.delete_chunks_by_article(cfg.qdrant_collection, chunk.article_title)
                    bar.update(1)  # advance one article

                batch_chunks.append(chunk)
                batch_texts.append(chunk.text)

                # Upsert in batches to avoid overwhelming Qdrant and to batch embeddings
                if len(batch_chunks) >= BATCH_SIZE:
                    # Embed all texts in a single API call
                    vectors = embedder.embed(batch_texts)
                    collection_mgr.upsert_chunks(cfg.qdrant_collection, batch_chunks, vectors)
                    total_chunks += len(batch_chunks)
                    batch_chunks = []
                    batch_texts = []

            # Final batch
            if batch_chunks:
                vectors = embedder.embed(batch_texts)
                collection_mgr.upsert_chunks(cfg.qdrant_collection, batch_chunks, vectors)
                total_chunks += len(batch_chunks)

    except IngesterError as e:
        click.echo(f"Error during ingestion: {e}", err=True)
        raise click.Abort()

    click.echo(f"Ingestion complete: {total_chunks} chunks from {total_articles} articles")
