# AGENTS.md — Wikipedia Rhizome

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
rhizome ingest --categories Modernism,Postmodernism --max-articles 500

# Run traversal
rhizome traverse "the tension between modernism and postmodernism" --depth 8

# Synthesize a thesis from a traversal (needs LLM_GATEWAY_URL)
rhizome idea "the tension between structure and event" -o thesis.md --save-material
```

## Key design decisions

- **No LLM prose in stitching** — `traverse` outputs original Wikipedia text with citations
- **`idea` is deliberately minimal** — one walk, one LLM call, one thesis in prose. No planner, no verification stage, no critic, no quality gate; the reader judges the output. All traversal knobs are exposed rather than auto-tuned.
- **The traversal engine is never steered mid-walk** — `idea` acts on it only through its inputs (seed, epsilon, temperature, depth, top_k) and by reading its output. Retrieval stays autonomous; judgement lives entirely in synthesis.
- **Seed is a lens, not a redirect** — verified by A/B: with heterogeneous walk material the seed decides which collision gets argued, but it cannot change the material. So `--epsilon`/`--depth` matter more than the seed phrasing.
- **`--save-material`** writes the walked fragments beside the thesis so a run stays traceable; the walk is non-deterministic, so without it a run is unreproducible once the process exits. Written before the LLM call so a synthesis failure does not discard the walk.
- **Pin a concrete model name, not a router alias** — aliases resolve server-side and can be silently re-pointed, so runs stop being comparable over time.
- **Embedder ABC** — swap HuggingFace for OpenAI/Anthropic by replacing `HuggingFaceEmbedder`
- **Chunk IDs** — `{article-slug}-{ordinal}` format, stable across re-ingests
- **Fallback bound** — 2+ consecutive fallback steps → forced random global jump

## Dependencies

- Python >= 3.11
- Qdrant (running locally at http://localhost:6333)
- OpenAI API key (set `OPENAI_API_KEY` env var) or HuggingFace API token (`HF_API_TOKEN`)
