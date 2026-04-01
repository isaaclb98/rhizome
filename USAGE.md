# Wikipedia Rhizome — Usage Guide

Rhizome traverses Wikipedia's vector space to pull content into writing. Think of it as semantic random-walk research: you give it a concept, it walks through semantically-related Wikipedia articles, and produces markdown with citations.

## Prerequisites

- Python 3.11+
- [Qdrant](https://qdrant.tech/documentation/quick-start/) running locally (or update `config.yaml` to point at a remote instance)
- OpenAI API key

## Installation

```bash
# Clone and install
cd wikipedia-modernity
python3.14 -m pip install -e .

# Or with uv
uv pip install -e .
```

## Configuration

Edit `config.yaml` in the project root. The key settings:

```yaml
openai:
  api_key: "${OPENAI_API_KEY}"   # your OpenAI key, or set OPENAI_API_KEY env var
  model: "text-embedding-3-small" # 1536-dim, ~$0.02/million tokens

vectorstore:
  collection: "modernity-v1"      # Qdrant collection name
  vector_size: 1536              # must match embedding model dimensions

corpus:
  source: "wikipedia"
  domains:                      # Wikipedia search terms to ingest
    - "Modernism"
    - "Postmodernism"
  max_articles: 500             # articles per domain

traversal:
  depth: 8                      # steps to walk through the vector space
  epsilon: 0.1                  # exploration probability (0.0 = greedy, 1.0 = random)
  top_k: 5                      # candidates to consider at each step
```

Set your OpenAI key:

```bash
export OPENAI_API_KEY=sk-...
```

## Ingest: Build the Corpus

First, ingest Wikipedia articles into Qdrant. This fetches articles via the Wikipedia API, chunks them into paragraphs, embeds them with OpenAI, and stores the vectors.

```bash
# Ingest the default domains from config.yaml
rhizome ingest

# Override domains from the command line
rhizome ingest --domain Modernism --domain Postmodernism

# Limit how many articles
rhizome ingest --max-articles 200

# Custom config path
rhizome ingest --config my-config.yaml
```

Ingest is idempotent. Running it twice will create duplicate chunks (the Qdrant upsert is batched at 50 chunks at a time, ~10-20 seconds per batch depending on article length).

## Traverse: Generate Narrative

Once you have a corpus, traverse it to produce a writing prompt or draft section.

```bash
# Basic traversal — prints markdown to stdout
rhizome traverse "the tension between modernism and postmodernism"

# Save output to a file
rhizome traverse "post-structuralist critique of meaning" -o output.md

# Override traversal parameters
rhizome traverse " Derrida and architecture" --depth 12 --epsilon 0.15 --top-k 7
```

The output is markdown with inline citations pointing back to Wikipedia articles and paragraphs. It is not a finished essay — it is research fodder with sources.

### Output format

Each paragraph is sourced:

```markdown
Postmodernism rejects the grand narratives of modernism, [1] treating
style as a cultural construct rather than an aesthetic absolute. The
emphasis shifts from permanence toplay, from the timeless to the
contingent.

[1] https://en.wikipedia.org/wiki/Postmodernism — "Postmodernism"
```

### Tuning traversal

| Flag | What it does |
|------|-------------|
| `--depth` | How many hops. Higher = longer/more tangential output, riskier |
| `--epsilon` | Exploration vs exploitation. 0.0 = always pick the nearest neighbor; 0.1-0.2 = occasionally jump somewhere random to escape local basins |
| `--top-k` | Candidate pool size at each step. Higher = more options but slower |

## How the traversal works

The traversal is an epsilon-greedy random walk through the vector space:

1. Embed the starting concept with OpenAI
2. Query Qdrant for the `top_k` nearest neighbors
3. With probability `epsilon`, pick a random candidate instead of the best match
4. If no unvisited neighbors remain, force a random global jump
5. Repeat until `depth` is reached

The "narrative" is stitched from the actual Wikipedia paragraph text, with citations. The LLM (you, writing with this as research) decides what to keep and how to connect the threads.

## Two-step workflow

```bash
# 1. Build the corpus (run once)
rhizome ingest --domain Modernism --domain Postmodernism --max-articles 500

# 2. Traverse as many times as you want
rhizome traverse "the death of the author and its implications"
rhizome traverse "Bauhaus influence on graphic design"
rhizome traverse "deconstruction as architectural theory"
```

## Troubleshooting

**"Collection not found"** — Run `rhizome ingest` first.

**"OPENAI_API_KEY environment variable is not set"** — Set the env var: `export OPENAI_API_KEY=sk-...`

**Rate limiting from OpenAI** — Ingest is sequential and slow. For large corpora, you can parallelize the embedding step, but that requires changes to `ingest.py`.

**Qdrant connection refused** — Make sure Qdrant is running. Default address is `localhost:6333`. Update `vectorstore:` in config.yaml to set `url: "http://localhost:6333"` or point at a remote instance.
