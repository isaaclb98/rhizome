"""Generate ideas by directing rhizomatic traversals with an LLM."""

from __future__ import annotations

import json
import sys

import click

from rhizome.config import get_config
from rhizome.ideas.agent import IdeaAgent, IdeaAgentError
from rhizome.ideas.renderer import render_cards
from rhizome.llm.gateway import GatewayEmbedder, GatewayLLM
from rhizome.vectorstore.client import VectorStoreClient


@click.command()
@click.argument("seed")
@click.option("--output", "-o", type=click.Path(), help="Markdown output file (default: stdout)")
@click.option("--json-output", "-j", type=click.Path(), help="Also write the full run as JSON")
@click.option("--max-walks", type=int, help="Cap on traversals per run (overrides config)")
@click.option("--walk-depth", type=int, help="Default steps per walk (overrides config)")
@click.option("--rounds", type=int, help="Synthesis rounds (overrides config)")
@click.option("--max-ideas", type=int, help="Concepts per round (overrides config)")
@click.option("--model", type=str, help="LLM model name (overrides config)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def idea(
    seed: str,
    output: str | None,
    json_output: str | None,
    max_walks: int | None,
    walk_depth: int | None,
    rounds: int | None,
    max_ideas: int | None,
    model: str | None,
    quiet: bool,
):
    """Generate concept ideas from a SEED via LLM-directed rhizomatic traversal.

    The LLM plans epsilon-greedy walks through the corpus, diagnoses the loose
    fragments each walk returns, re-walks with different knobs, then forges
    concept cards with provenance-tagged evidence and vets them against the
    corpus.
    """
    cfg = get_config()

    if not cfg.gateway_base_url:
        click.echo(
            "Error: LLM_GATEWAY_URL is not set. The idea agent needs an "
            "OpenAI-compatible endpoint serving both chat completions and "
            "embeddings. Set it in .env or the environment.",
            err=True,
        )
        sys.exit(2)

    def progress(stage: str, message: str) -> None:
        if not quiet:
            click.echo(f"[{stage}] {message}", err=True)

    llm = GatewayLLM(
        base_url=cfg.gateway_base_url,
        model=model or cfg.llm_model,
        api_key=cfg.gateway_api_key,
    )
    embedder = GatewayEmbedder(
        base_url=cfg.gateway_base_url,
        model=cfg.embedding_model,
        api_key=cfg.gateway_api_key,
    )
    vector_store = VectorStoreClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection_name=cfg.qdrant_collection,
    )

    agent = IdeaAgent(
        llm=llm,
        embedder=embedder,
        vector_store=vector_store,
        collection_name=cfg.qdrant_collection,
        max_walks=max_walks if max_walks is not None else cfg.idea_max_walks,
        walk_depth=walk_depth if walk_depth is not None else cfg.idea_walk_depth,
        max_rounds=rounds if rounds is not None else cfg.idea_max_rounds,
        max_ideas=max_ideas if max_ideas is not None else cfg.idea_max_ideas,
        on_progress=progress,
    )

    try:
        run = agent.run(seed)
    except IdeaAgentError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    markdown = render_cards(run)
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(markdown)
        progress("done", f"cards written to {output}")
    else:
        click.echo(markdown)

    if json_output:
        with open(json_output, "w", encoding="utf-8") as handle:
            json.dump(run.to_dict(), handle, indent=2, ensure_ascii=False)
        progress("done", f"run JSON written to {json_output}")

    kept = run.kept_cards()
    progress(
        "done",
        f"{len(run.walks)} walks, {len(run.cards)} concepts, {len(kept)} kept",
    )
