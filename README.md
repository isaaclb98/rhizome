# Wikipedia Rhizome

Rhizome walks through Wikipedia's vector space and pulls content into writing. You give it a concept, it traverses semantically-related Wikipedia articles using epsilon-greedy search, and outputs markdown with citations. No LLM required — just embeddings and the original Wikipedia text.

See it as semantic random-walk research: instead of keyword search, you traverse a vector space of Wikipedia paragraphs and pull out what you find along the path.

## TL;DR

```bash
# 1. Set your OpenAI key
export OPENAI_API_KEY=sk-...

# 2. Install
uv pip install -e .

# 3. Build the corpus (run once)
rhizome ingest --domain Modernism --domain Postmodernism --max-articles 20000

# 4. Traverse (run as many times as you want)
rhizome traverse "the tension between modernism and postmodernism" -o draft.md
```

---

## First-time setup

### 1. Get an OpenAI API key

Sign up at [platform.openai.com](https://platform.openai.com) if you don't have one. Rhizome uses the `text-embedding-3-small` model ($0.02/million tokens — cheap for a 500-article corpus).

**Option A — `.env` file (recommended):**
```bash
cp .env.example .env
# Then edit .env and add your keys
```

**Option B — environment variables:**
```bash
export OPENAI_API_KEY=sk-...
export QDRANT_API_KEY=...  # only if using cloud Qdrant
```

Add env vars to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to persist them.

### 2. Install

Requires Python 3.11 or later.

**With uv (recommended):**
```bash
uv pip install -e .
```

**With pip (system Python — requires Ubuntu/Debian workaround):**
```bash
# If you get "externally-managed-environment", use --break-system-packages
pip install -e . --break-system-packages
```

**Or with a virtual environment:**
```bash
python3.14 -m venv .venv && source .venv/bin/activate && pip install -e .
```

### 3. Qdrant (already configured)

The `config.yaml` points at a remote Qdrant instance (`https://qdrant.aizaku.ca`). No Docker or local setup needed. If you want to run Qdrant locally instead:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Then update `config.yaml`:

```yaml
vectorstore:
  url: "http://localhost:6333"
  api_key: null  # no key needed for local
```

---

## Ingest: Build the corpus

Ingest fetches Wikipedia articles, chunks them into paragraphs, embeds each chunk with OpenAI, and stores the vectors in Qdrant.

```bash
# Ingest specific domains
rhizome ingest --domain Modernism --domain Postmodernism --max-articles 20000

# Override settings from the command line
rhizome ingest --domain Modernism --max-articles 200
```

This takes a few minutes for 500 articles (OpenAI rate limits are the bottleneck). Progress is printed to stdout.

**How it works:**
1. PetScan looks up all Wikipedia articles in the category tree under each domain. The domains are combined as a **union** — articles from either category are included.
2. Each article is fetched from the Wikipedia API
3. Articles are chunked at paragraph boundaries (paragraphs over 500 chars are split at sentence boundaries)
4. Bibliography and reference sections are stripped before chunking
5. Each chunk is embedded with OpenAI `text-embedding-3-small` (1536 dimensions)
6. Chunks are upserted to Qdrant in batches of 10

**What to expect:**
- 500 articles → ~2,500 chunks → ~2,500 OpenAI embedding calls
- Total ingest time is roughly 10-15 minutes for 500 articles

---

## Traverse: Walk the vector space

Once you have a corpus, traverse it to generate research material.

```bash
# Basic traversal — prints markdown to stdout
rhizome traverse "the tension between modernism and postmodernism"

# Save output to a file
rhizome traverse "post-structuralist critique of meaning" -o output.md

# Override traversal parameters
rhizome traverse "Derrida and architecture" --depth 12 --epsilon 0.15 --top-k 7
```

The output is not a finished essay. It is a sequence of Wikipedia paragraphs, ordered by semantic similarity to the traversal path, with citations.

### Tuning traversal

| Flag | Default | What it does |
|------|---------|-------------|
| `--depth` | 8 | How many hops. Higher = longer/more tangential output |
| `--epsilon` | 0.1 | Exploration probability. 0.0 = always pick the nearest neighbor; 0.2 = 20% random jumps |
| `--top-k` | 5 | Candidate pool size. Higher = more options but slower per step |

**How the traversal works:**
1. Embed the starting concept with OpenAI
2. Query Qdrant for the `top_k` nearest chunks
3. With probability `epsilon`, pick a random candidate (explore)
4. With probability `1-epsilon`, pick the nearest (exploit)
5. If the nearest neighbor is already visited, fall back to the next-nearest
6. After 2 consecutive fallbacks (stuck in local basin), force a random global jump to escape
7. Repeat until `depth` is reached

---

## Two-step workflow

```bash
# Step 1: Build the corpus (once)
rhizome ingest --domain Modernism --domain Postmodernism --max-articles 20000

# Step 2: Traverse as many times as you want (each run is ~30 seconds)
rhizome traverse "the death of the author and its implications"
rhizome traverse "Bauhaus influence on graphic design"
rhizome traverse "deconstruction as architectural theory"
```

---

## Output format

Each paragraph is followed by its citation:

```markdown
Postmodernism rejects the grand narratives of modernism, treating
style as a cultural construct rather than an aesthetic absolute.

*[Source: Postmodernism](https://en.wikipedia.org/wiki/Postmodernism)*
```

The file footer shows the traversal path length and tool name.

---

## Troubleshooting

**"OPENAI_API_KEY environment variable is not set"**
Set the env var: `export OPENAI_API_KEY=sk-...`

**"Collection not found"**
Run `rhizome ingest` first to build the corpus.

**"Connection refused" or Qdrant errors**
Check that Qdrant is running. The remote instance at `https://qdrant.aizaku.ca` requires network access. For local: `docker run -p 6333:6333 qdrant/qdrant`.

**Rate limiting from OpenAI**
Ingest is sequential. For large corpora, embedding is the bottleneck. A 500-article ingest takes roughly 10-15 minutes.

---

## Configuration reference

All settings live in `config.yaml`:

```yaml
openai:
  api_key: "${OPENAI_API_KEY}"   # env var or literal key
  model: "text-embedding-3-small" # 1536-dim embeddings

vectorstore:
  url: "https://qdrant.aizaku.ca" # remote Qdrant
  api_key: "${QDRANT_API_KEY}"     # cloud Qdrant key
  collection: "modernity-v1"       # collection name
  vector_size: 1536                # must match embedding model

corpus:
  domains:
    - "Modernism"       # Wikipedia categories to ingest
    - "Postmodernism"
  max_articles: 500      # per domain
  depth: 1               # PetScan category tree depth (1 = top-level only)

traversal:
  depth: 8               # number of hops per traversal
  epsilon: 0.1           # exploration probability
  top_k: 5               # candidates per step
```
