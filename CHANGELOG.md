# Changelog

## 0.4.0 (2026-04-03)

### Features
- **Web visualizer** — FastAPI + React/D3 web app for traversing and visualizing Wikipedia embeddings as a force-directed graph. Serves both API and frontend from a single Docker image.
- **Domain field in chunks** — Each chunk now stores its Wikipedia domain (Modernism, Postmodernism, Critical theory) in Qdrant. Used for domain-colored nodes in the visualizer.
- **POST /traverse endpoint** — New REST API endpoint for running traversals. Returns path with `domain`, `similarity`, and `forced_jump` fields per step.
- **SPA routing** — FastAPI serves `index.html` for non-API routes, enabling client-side routing and browser refresh support.
- **Domain migration CLI** — `rhizome migrate-domain-field` command patches existing Qdrant collections to add the `domain` field to existing points.

### Breaking Changes
- **Chunk dataclass** — `chunk_article()` now requires a `domain` parameter. Update any code that calls it directly.
- **TraversalStep** — Now includes `domain`, `similarity`, and `forced_jump` fields. Update any code that unpacks `TraversalStep` objects.

### Changes
- **Dockerfile** — Replaced CLI-only image with multi-stage build (Node + Python) that serves both the FastAPI web app and the React frontend from a single image.
- **WikipediaIngester** — Now tracks which domain each article was discovered under and passes it to chunks.

## 0.3.0 (2026-04-02)

### Bug Fixes
- **pydantic-settings dependency** — Added to `pyproject.toml` dependencies (was imported but not declared)
- **Qdrant point ID format** — Slug-based chunk IDs like `Symbolic-annihilation-001` are now hashed to UUIDs before use as Qdrant point IDs (Qdrant requires UUIDs or unsigned integers)
- **Config defaults** — `embedder_type`, `qdrant_collection`, and `top_k` now have sensible defaults so `rhizome traverse` works without all env vars set
- **Docker compose env vars** — `docker-compose.yml` and `docker-compose.dev.yml` now pass `QDRANT_COLLECTION` and `EMBEDDER_TYPE` to the container

### Infrastructure
- **Docker layer caching** — `.github/workflows/docker-push.yml` pushes image to DockerHub with layer caching on every push to main
- **Dockerfile cleanup** — Removed stale `config.yaml` copy and conflicting `entrypoint` from both compose files

## 0.2.0 (2026-04-02)

### Features
- **Pydantic config module** — All settings (QDRANT_URL, QDRANT_COLLECTION, EMBEDDER_TYPE, HF_API_TOKEN, etc.) validated via Pydantic with env var binding
- **Embedder factory** — `get_embedder()` supports OpenAI and HuggingFace Inference API providers with consistent interface
- **Stored-vector traversal** — Traversal engine retrieves stored vectors from Qdrant instead of re-embedding, reducing API calls and improving semantic continuity

### Bug Fixes
- **Traversal re-embed bug** — `engine.py` now uses `selected["vector"]` when available, falls back to re-embed only when stored vector is missing
- **Slug collision** — Chunk IDs preserve case in slugs, preventing collision between articles differing only by case (e.g. 'Modernism' vs 'modernism')
- **Forced jump with_vector** — Global forced jumps pass `with_vector=False` since only IDs are needed for random selection

### Improvements
- **Batch OpenAI embedding** — Ingest batches up to 100 texts per API call, respecting the 2048-input batch limit
- **delete_chunks_by_article optimization** — Single delete-by-filter call instead of scroll-paginated deletes
- **search_excluding query_filter** — Added optional `query_filter` parameter for pre-filtering before exclusion
- **EmbeddingError deduplication** — Moved to `embedder/base.py`, imported by `openai.py` and `huggingface.py`

### Refactor
- **Chunk ID migration** — Chunk IDs changed from UUID format to stable slug-based `{article-slug}-{ordinal}` format
- **Config singleton** — `get_config()` returns a singleton `RhizomeConfig` instance
