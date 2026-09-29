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
  llm/             — LLM client interface + OpenAI-compatible gateway transport
  vectorstore/     — Qdrant client wrapper + collection management
  traversal/       — Epsilon-greedy traversal engine
  ideas/           — Idea agent: LLM-directed traversal + concept synthesis
  stitching/       — Markdown formatter with citations
  cli/             — Click CLI commands

.env.example       — Environment variables template
pyproject.toml     — Python package definition
IDEAS.md           — Idea agent design (loop, provenance, verification)
```

## Commands

```bash
# Ingest Wikipedia articles
pip install -e .
rhizome ingest --categories Modernism,Postmodernism --max-articles 500

# Run traversal
rhizome traverse "the tension between modernism and postmodernism" --depth 8

# Generate concept ideas (LLM-directed traversal)
rhizome idea "the tension between structure and event" -o cards.md -j run.json
```

## Key design decisions

- **No LLM prose in v1** — stitching only, original Wikipedia text
- **Embedder ABC** — swap HuggingFace for OpenAI/Anthropic by replacing `HuggingFaceEmbedder`
- **Chunk IDs** — `{article-slug}-{ordinal}` format, stable across re-ingests
- **Fallback bound** — 2+ consecutive fallback steps → forced random global jump
- **Idea agent leaves the traversal engine untouched** — the LLM acts on it only through its inputs (seed, epsilon, temperature, depth, top_k) and by reading its output. No mid-walk steering.
- **Two-register provenance** — every idea component is tagged `corpus` (must cite a real chunk id, verbatim quote) or `model` (LLM training). Fabricated citations are demoted, never kept silently.
- **Verification both directions** — cards are embedded and searched to catch tracings; model-register claims are audited against retrieved passages.

## Dependencies

- Python >= 3.11
- Qdrant (running locally at http://localhost:6333)
- OpenAI API key (set `OPENAI_API_KEY` env var) or HuggingFace API token (`HF_API_TOKEN`)
