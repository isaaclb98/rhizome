# Idea Agent — DESIGN

An LLM agent that generates philosophical concepts by directing rhizomatic
traversal of a Wikipedia corpus. Built on rhizome's existing traversal engine,
which is used unmodified as a tool.

## The premise

Rhizome's traversal is already a bisociation machine. Three mechanisms produce
collisions between distant material:

- **explore step** — softmax-sampled from top-K rather than greedy, so the walk
  takes lateral moves
- **fallback** — top-K exhausted, pushed outward to the next unvisited chunk
- **forced global jump** — 2+ consecutive fallbacks, teleported to a random
  chunk unrelated to the current position

A greedy walk finds confirmation. A rhizome walk finds *discontinuity*. Every
jump has a defined before and after, with cosine similarity recorded, so
distance is measurable rather than guessed.

## Division of labour

The traversal engine stays autonomous and deliberately dumb. It runs
epsilon-greedy, returns a loose and disjointed fragment set, and applies no
judgement to what it finds. A single walk's output is a *sample*, not a result.

The LLM sits outside it. This is the load-bearing architectural decision:
**no mid-walk steering.** The engine is untouched, so the agent can only act on
it by choosing its inputs — the seed phrase and the traversal knobs — and by
reading its output.

Each side does what it is uniquely good at:

- The **corpus** is the anchor and the auditor. It keeps every claim traceable
  to real text and catches the model's overreach.
- The **model** is the connector and the navigator. It has read essentially all
  of continental philosophy, its critics, its secondary literature, and
  everything adjacent. It knows why a collision between Althusser and Bergson
  matters — knowledge no individual chunk carries.

Neither half alone is the tool. That is the RAG at the core: retrieval is not
the product, it is the grounding discipline that lets the model's whole
philosophical training loose on a seed without it floating away.

## The loop

```
seed
  |
  v
plan walks -------------------> LLM unpacks the seed from its own training:
  |                             sub-tensions, naive framings, and the retrieval
  |                             queries worth running
  v
walk(seed, knobs) ------------> TraversalEngine.traverse() -- unchanged
  |                             returns loose fragments + jump metadata
  v
assess ------------------------> what collided, what was dead weight,
  |                             which region was only grazed
  v
refine? -----------------------> re-walk with different knobs, or re-seed
  |                             (budget-capped; planner may decline)
  v
synthesize ---------------------> forge concept cards from ALL fragments
  |                               gathered so far
  v
novelty check ------------------> embed each card, search the corpus with it
  v
audit --------------------------> check model-register claims against corpus
  v
critique -----------------------> referee: map or tracing?
  v
more rounds? -------------------> re-seed a walk from the best survivor
  v
summarize ----------------------> closing assessment
```

### Traversal knobs are the LLM's action space

`epsilon`, `depth`, `temperature`, `top_k`, and the seed phrase are not user
CLI flags here — they are what the model adjusts when it diagnoses a bad walk.
That is what "the LLM improves upon traversal" means mechanically:

| Symptom | Correction |
|---|---|
| Too coherent, one tradition, no collisions | raise epsilon and temperature; seed from something foreign |
| Too scattered, nothing connects | lower temperature; shorten depth; seed more specifically |
| Grazed a productive region | seed directly from a concept inside it |

## Two-register provenance

The defining discipline. Every component of a forged concept is tagged:

- **corpus** — grounded in a retrieved chunk. Must cite a `chunk_id` that
  actually appears in the walk results, with a verbatim quote. The agent
  overwrites the model's claimed title/URL/similarity with the real source's
  values.
- **model** — drawn from training. Lineages, rival readings, what a concept was
  a reaction against, what it suppressed, which adjacent field names the thing.

A concept built only from corpus fragments is a collage. One built only from
training is ungrounded. The interesting ones are welded, and the weld is
visible. This is what makes "the LLM does more" safe: you always know which
half of a claim to trust, and how far.

### Grounding enforcement

Fabricated citations are worse than absent ones, so they are demoted rather
than kept. A corpus-register component citing an unknown `chunk_id` becomes
model-register, annotated `uncorroborated`, with the reason recorded. A quote
that is not verbatim in the cited chunk is replaced with the nearest verbatim
excerpt from it.

## Verification

**Novelty check (the tracing test).** Embed the card and search the corpus with
it. High `max_similarity` means the corpus already contains that synthesis —
the model is paraphrasing the literature back, which is a *tracing*. Low
similarity means the card occupies a void, which is a *map*. Novelty against
intellectual history becomes a measurable cosine distance rather than a vibe.

| `max_similarity` | verdict |
|---|---|
| >= 0.72 | `tracing` |
| >= 0.60 | `borderline` |
| < 0.60 | `novel` |

**Claim audit.** RAG checking the parametric channel, not just grounding it.
Every model-register claim is embedded and searched; retrieved passages are
passed to the critic with the claim, which rules `corroborated`, `contested`,
or `uncorroborated`. Contested claims flag their card.

**Critique.** A hostile referee distinguishes map from tracing, asks whether
components genuinely require one another, and whether an occupied void is
*interesting* empty space or merely incoherent empty space. `discard` verdicts
cap confidence at 0.2; naming an `is_tracing_of` target caps it at 0.3. Referee
reasoning is appended to the card's objection field, so the strongest argument
against is always on record.

Caps must never silently overwrite the referee's own ruling. A card can be
ruled `keep` *and* capped for being a tracing, and both signals are retained:
`verdict` and `tracing_of` are stored on the card, and `confidence_note` records
the referee's original score alongside the reason for the cap. This was a real
defect found in the first end-to-end run — every card scored exactly 0.30 with
no recorded cause, because a non-empty `is_tracing_of` was capping confidence
while the `keep` verdict and the 0.95 score were discarded. In a system whose
whole premise is auditability, the critique stage discarding its own ruling was
the worst possible place for that failure.

Survivor selection for later rounds excludes `discard` verdicts and tracings —
re-seeding from a restatement of existing work walks back into territory the
corpus already covers.

## Rounds

Later rounds re-seed a walk from the best surviving concept, so the next walk
starts somewhere the previous one earned. Survivor selection prefers
uncontested cards at confidence >= 0.5. If nothing survives, the run closes
rather than burning a walk on a concept the referee already rejected.

## Failure handling

Per-stage failures are logged and skipped so a partial run still returns usable
material; only a dead planner or a corpus that yields no fragments at all
abort. Walk failures are recorded on the walk with the error as its note.

Reasoning models need specific handling, learned from real runs:

- **thinking blocks** — reasoning models prepend content that is not the
  payload. It is stripped before JSON extraction, including unterminated blocks
  from mid-reasoning truncation.
- **truncation** — `finish_reason == "length"` means the budget was spent,
  usually on reasoning rather than the answer. This raises
  `TruncatedResponseError`, and `complete_json` doubles the budget (capped at
  32768) and retries from the original messages rather than appending a
  correction, since the failure was capacity, not comprehension.

## Corpus caveat

The live corpus is **general Wikipedia**, not a philosophy-only collection.
Bare technical terms get hijacked by their dominant everyday sense: "phase
transition" retrieves physics and narrative criticism; "structure" retrieves
buildings; "fold" retrieves paper and anatomy. The planner prompt therefore
requires domain-qualified seeds ("structural causality in Althusser", not
"structure"). This is a prompt-level mitigation, not a solved problem.

A structural fix would be to filter retrieval to philosophy chunks. That is not
currently possible against the live `rhizome` collection: sampled payloads carry
only `id`, `text`, `article_title`, and `article_url` — there is **no `domain`
field**, despite `rhizome/migrations/add_domain_field.py` existing. Backfilling
domains (or ingesting a philosophy-scoped collection) would let
`search_excluding`'s `query_filter` constrain walks to the intended register.

## Layout

```
rhizome/llm/
  base.py       LLMClient protocol; LLMError, TruncatedResponseError
  gateway.py    OpenAI-compatible transport; thinking-block strip; JSON extraction
rhizome/ideas/
  models.py     IdeaRun, IdeaCard, ComponentEvidence, Fragment, WalkPlan/Result
  prompts.py    planner, synthesizer, critic, summarizer, audit prompts
  agent.py      the loop: plan -> walk -> assess -> synthesize -> verify -> critique
  renderer.py   markdown concept cards + traversal trail
rhizome/cli/commands/idea.py   the `idea` command
```

## Config

Env vars, following the project's `hardcoded < .env < CLI flag` precedence.

| Var | Default | Meaning |
|---|---|---|
| `LLM_GATEWAY_URL` | unset (required) | OpenAI-compatible endpoint |
| `LLM_GATEWAY_API_KEY` | unset | Optional bearer token |
| `LLM_MODEL` | `auto/best-reasoning` | Chat model |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Must match the corpus's embedding model |
| `IDEA_MAX_WALKS` | `3` | Hard cap on traversals |
| `IDEA_WALK_DEPTH` | `8` | Default steps per walk |
| `IDEA_MAX_ROUNDS` | `2` | Synthesis rounds |
| `IDEA_MAX_IDEAS` | `3` | Concepts requested per round |

The embedding model must produce vectors matching the collection's dimension
(1536 for the live `rhizome` collection). A mismatch fails at query time, not
config time.

## Usage

```bash
rhizome idea "the tension between structure and event" \
  --max-walks 3 --walk-depth 6 --rounds 2 --max-ideas 3 \
  -o cards.md -j run.json
```

`-o` writes concept cards as markdown; `-j` dumps the full run (every walk,
every fragment, every card with its evidence trail) as JSON for auditing.
Progress goes to stderr, so stdout stays clean for piping.

## Deliberately not built

- **Mid-walk steering.** Would require engine changes and collapse the
  retrieval/generation separation.
- **Problem-mode** (`--mode problem`). Same pipeline, different stage-3 prompt;
  a flag, not a project.
- **Output re-ingestion.** Feeding generated cards back as a second collection
  would let future walks collide with past ideas. Deferred, but it is why cards
  are chunk-shaped from the start.
- **Domain filtering.** Would fix the homonym-hijack problem structurally, but
  the live collection carries no `domain` field, so it needs a backfill or a
  philosophy-scoped re-ingest first. `search_excluding` already accepts a
  Qdrant filter, so the retrieval side needs no change.
