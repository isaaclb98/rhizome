# CLAUDE.md — Wikipedia Rhizome

## What this project is

Rhizome is a CLI tool for rhizomatic traversal of Wikipedia embeddings. It walks through a semantic vector space using epsilon-greedy search and produces a navigable prose document from the path taken.

## Project structure

```
rhizome/
  config.py        — Pydantic config (env vars: QDRANT_URL, HF_API_TOKEN, etc.)
  corpus/          — Wikipedia ingestion + chunking
  embedder/        — Embedding provider interface (OpenAI, HuggingFace)
  embedder/factory.py — Embedder factory (get_embedder)
  vectorstore/     — Qdrant client wrapper + collection management
  traversal/       — Epsilon-greedy traversal engine
  stitching/       — Markdown formatter with citations
  cli/             — Click CLI commands

.env.example       — Environment variables template
pyproject.toml     — Python package definition
```

## Commands

```bash
# Ingest Wikipedia articles
pip install -e .
rhizome ingest --domain Modernism --max-articles 500

# Run traversal
rhizome traverse "the tension between modernism and postmodernism" --depth 8
```

## Key design decisions

- **No LLM prose in v1** — stitching only, original Wikipedia text
- **Embedder ABC** — swap HuggingFace for OpenAI/Anthropic by replacing `HuggingFaceEmbedder`
- **Chunk IDs** — `{article-slug}-{ordinal}` format, stable across re-ingests
- **Fallback bound** — 2+ consecutive fallback steps → forced random global jump

## Dependencies

- Python >= 3.11
- Qdrant (running locally at http://localhost:6333)
- OpenAI API key (set `OPENAI_API_KEY` env var) or HuggingFace API token (`HF_API_TOKEN`)
