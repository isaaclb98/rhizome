# AGENTS.md — Wikipedia Rhizome

## What this project is

Rhizome is a CLI tool for rhizomatic traversal of Wikipedia embeddings. It walks through a semantic vector space using epsilon-greedy search and produces a navigable prose document from the path taken.

## Project structure

```
rhizome/
  config.py        — Pydantic config (env vars: QDRANT_URL, EMBEDDER_TYPE, etc.)
                     Reads the repo .env via pydantic-settings `env_file`.
  corpus/          — Wikipedia ingestion + chunking
  embedder/        — Embedding provider interface (OpenAI, HuggingFace)
  embedder/factory.py — Embedder factory (get_embedder: openai|huggingface|gateway)
  gateway.py       — OpenAI-compatible gateway clients (GatewayLLM, GatewayEmbedder)
  vectorstore/     — Qdrant client wrapper + collection management
  traversal/       — Epsilon-greedy traversal engine
  stitching/       — Markdown formatter with citations
  api/main.py      — FastAPI app (/traverse, /idea, static frontend)
  visualizer/app/  — Vite + React frontend (Synthesize and Traverse tabs)
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

# Serve the API + visualizer locally
cd rhizome/visualizer/app
VITE_OUTPUT_DIR=$PWD/dist npm run build
RHIZOME_STATIC_DIR=$PWD/dist uvicorn rhizome.api.main:app --host 127.0.0.1 --port 8765
```

The API reads `.env` from the repo root, so no extra env plumbing is needed for
local runs. Embedding is optional: if the embedder cannot be built at startup
(missing `OPENAI_API_KEY` with `EMBEDDER_TYPE=openai`), the server still comes up
and embedding-dependent routes return 503. Set `EMBEDDER_TYPE=gateway` to embed
through `LLM_GATEWAY_URL` without any key.

## API routes

- `POST /traverse` — walk, return path + stitched material
- `POST /traverse/stream` — same, as SSE `step` events
- `POST /idea` — walk then synthesize; returns `{thesis, path, stats}`
- `POST /idea/stream` — SSE: `step` events during the walk, one `thesis` event,
  then `done` with stats. The LLM call itself is not token-streamed.

`/idea` shares its helpers with the CLI (`run_traversal`, `synthesize_thesis`,
`compute_stats` in `rhizome/cli/commands/idea.py`) so the two paths cannot drift.
A per-request `llm_model` overrides the configured model.

## Key design decisions

- **No LLM prose in stitching** — `traverse` outputs original Wikipedia text with citations
- **`idea` is deliberately minimal** — one walk, one LLM call, one thesis in prose. No planner, no verification stage, no critic, no quality gate; the reader judges the output. All traversal knobs are exposed rather than auto-tuned.
- **Route order in `api/main.py` is load-bearing** — Starlette matches in registration order, and a `StaticFiles` mount matches every path. Any mount at `/` declared mid-module silently swallows later API routes, answering POSTs with `{"detail":"Method Not Allowed"}` that looks exactly like a routing miss. Keep static mounts last and scoped to `/assets`. `TestRouteOrdering` in `tests/test_api.py` enforces this.
- **Embedding is optional** — a failed embedder at startup logs and continues; routes that need it return 503. This lets the API serve the visualizer and the `/idea` endpoints on machines with no embedding credentials.
- **The traversal engine is never steered mid-walk** — `idea` acts on it only through its inputs (seed, epsilon, temperature, depth, top_k) and by reading its output. Retrieval stays autonomous; judgement lives entirely in synthesis.
- **Assertive voice is a hard constraint** — theses carry no first person in any form and no hedging or process narration ("I assume", "it seems", "arguably"). Claims are stated as fact; thin material is described as a property of the material, not as uncertainty. The prompt must not ask the model to narrate which claims came from the fragments versus its own knowledge — that invitation is what produces the reflective voice.
- **The synthesis prompt is domain-neutral** — it never names Wikipedia or implies a discipline, even though the bundled ingest pipeline is Wikipedia-specific. Corpus provenance is the ingest stage's business; synthesis works against whatever collection it is pointed at. `test_prompt_is_domain_neutral` scans the template (not a rendered prompt, since fragments legitimately carry corpus URLs) to keep it that way.
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
