"""Migrate existing Qdrant chunks to add the domain field.

This script handles collections created before the domain field was added in v0.4.0.
It scrolls all points, determines each article's Wikipedia domain using PetScan
category membership, patches the domain field, and upserts the points back.

Usage:
    # As a CLI command (after installing rhizome):
    rhizome migrate-domain-field --collection modernity-v1

    # Or run directly:
    python -m rhizome.migrations.add_domain_field --collection modernity-v1

Prerequisites:
    - QDRANT_URL, QDRANT_COLLECTION env vars must be set
    - EMBEDDER_TYPE, OPENAI_API_KEY (or HF_API_TOKEN) for embedding API access
"""

import argparse
import sys
from typing import Iterator

from rhizome.config import get_config
from rhizome.vectorstore.collection import CollectionManager
from rhizome.vectorstore.client import VectorStoreClient
from rhizome.embedder.factory import get_embedder

PETSCAN_URL = "https://petscan.wmflabs.org/"


def _http_get(url: str, params: dict, timeout: int = 30) -> dict:
    import requests
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def discover_articles_by_domain(domains: list[str], depth: int = 1) -> dict[str, str]:
    """Query PetScan for each domain separately and build article→domain mapping.

    Articles appearing in multiple domains are assigned to the first domain
    that returned them (consistent ordering via domains list).
    """
    article_domains: dict[str, str] = {}

    for domain in domains:
        params = {
            "language": "en",
            "project": "wikipedia",
            "depth": depth,
            "categories": domain,
            "combination": "union",
            "format": "json",
            "doit": 1,
        }
        data = _http_get(PETSCAN_URL, params)
        articles = data.get("*", [{}])[0].get("a", {}).get("*", [])
        for article in articles:
            title = article["title"].replace("_", " ")
            if title not in article_domains:
                article_domains[title] = domain

    return article_domains


def scroll_all_points(
    client: "QdrantClient",
    collection_name: str,
    batch_size: int = 100,
) -> Iterator[list[dict]]:
    """Scroll all points in a collection and yield batches."""
    from qdrant_client.models import ScrollFilter

    offset = None
    while True:
        result = client.scroll(
            collection_name=collection_name,
            scroll_filter=ScrollFilter(),
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points = result.points
        if not points:
            break
        yield [
            {"id": p.id, "payload": p.payload, "vector": p.vector}
            for p in points
        ]
        offset = result.next_page_offset


def migrate_domain_field(
    collection_name: str,
    domains: list[str] | None = None,
) -> tuple[int, int]:
    """Migrate domain field for all chunks in a Qdrant collection.

    Args:
        collection_name: Name of the Qdrant collection.
        domains: Wikipedia domains to use for mapping. Defaults to config domains.

    Returns:
        Tuple of (total_points, migrated_points).
    """
    cfg = get_config()
    domains = domains or cfg.wikipedia_domains

    collection_mgr = CollectionManager(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
    vector_store = VectorStoreClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection_name=collection_name,
    )

    if not collection_mgr.collection_exists(collection_name):
        raise ValueError(f"Collection '{collection_name}' not found")

    print(f"Discovering article→domain mapping for {len(domains)} domains...")
    article_domains = discover_articles_by_domain(domains)
    print(f"  Found {len(article_domains)} unique articles across domains")

    # Check how many existing points are missing domain
    total = 0
    missing_domain = 0
    for batch in scroll_all_points(
        collection_mgr.client, collection_name
    ):
        for point in batch:
            total += 1
            if "domain" not in (point["payload"] or {}) or not point["payload"].get("domain"):
                missing_domain += 1

    print(f"  Total points: {total}")
    print(f"  Missing domain: {missing_domain}")

    if missing_domain == 0:
        print("All points already have domain field — nothing to do.")
        return total, 0

    print("Patching domain field for existing points...")

    # We need the embedder to re-embed (to get vectors for upsert).
    # But upsert requires vectors. If stored vectors exist, we have them.
    # The Qdrant client should return stored vectors with the points.
    # If with_vectors=False was used above, we need to re-fetch with vectors.
    # Instead: scroll with vectors, update payload, upsert with original vectors.

    embedder = get_embedder(
        embedder_type=cfg.embedder_type,
        openai_api_key=cfg.openai_api_key,
        hf_api_token=cfg.hf_api_token,
        hf_model=cfg.hf_model,
    )

    migrated = 0
    for batch in scroll_all_points(collection_mgr.client, collection_name):
        patched = []
        for point in batch:
            payload = point["payload"] or {}
            article_title = payload.get("article_title", "")
            domain = article_domains.get(article_title, "Unknown")
            if not payload.get("domain"):
                payload["domain"] = domain
                patched.append(
                    {
                        "id": point["id"],
                        "payload": payload,
                        "vector": point["vector"],  # may be None if stored without vector
                    }
                )
                migrated += 1

        if patched:
            # Upsert each point individually since vectors may differ
            for patched_point in patched:
                from qdrant_client.models import PointStruct

                if patched_point["vector"] is not None:
                    pt = PointStruct(
                        id=patched_point["id"],
                        vector=patched_point["vector"],
                        payload=patched_point["payload"],
                    )
                else:
                    # No stored vector — can't upsert with vector=None,
                    # so re-embed from the text
                    text = patched_point["payload"].get("text", "")
                    if text:
                        vec = embedder.embed([text])[0]
                        pt = PointStruct(
                            id=patched_point["id"],
                            vector=vec,
                            payload=patched_point["payload"],
                        )
                    else:
                        print(
                            f"  WARNING: point {patched_point['id']} has no text, skipping"
                        )
                        continue

                collection_mgr.client.upsert(
                    collection_name=collection_name,
                    points=[pt],
                )

        print(f"  Migrated {migrated}/{missing_domain} points...", end="\r")

    print(f"\nDone. Migrated {migrated} points.")
    return total, migrated


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant chunks to add domain field"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: from QDRANT_COLLECTION env var)",
    )
    args = parser.parse_args()

    collection = args.collection or get_config().qdrant_collection

    try:
        total, migrated = migrate_domain_field(collection)
        print(f"Result: {migrated}/{total} points updated")
        sys.exit(0)
    except Exception as e:
        print(f"Migration failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
