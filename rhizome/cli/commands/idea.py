"""Synthesize a thesis from a traversal's material."""

from pathlib import Path

import click

from rhizome.config import RhizomeConfig, get_config
from rhizome.embedder import Embedder, EmbeddingError
from rhizome.gateway import GatewayEmbedder, GatewayError, GatewayLLM
from rhizome.traversal.config import TraversalConfig
from rhizome.traversal.engine import TraversalEngine
from rhizome.traversal.engine import TraversalError, TraversalStep
from rhizome.vectorstore.client import VectorStoreClient
from rhizome.vectorstore.collection import CollectionManager

PROMPT = """You are given fragments collected by a random walk through a vector space of Wikipedia. They are deliberately disjointed — some will be unrelated to each other and to the seed. That is the point.

Write one thesis: a single argument with a real claim, built by synthesizing this material. Use the fragments as evidence and as raw material. Where you need connective tissue the fragments do not supply — lineage, framing, a concept you know — supply it from your own knowledge and say which you are doing.

Do not summarize the fragments in order. Do not list ideas. Do not comment on the traversal. Argue one thing, and let the collisions in the material carry it. Cite the articles you actually used.

Seed: {seed}

Fragments, in walk order:

{fragments}
"""


def format_fragments(path) -> str:
    """Render a traversal path as numbered material for the prompt.

    Args:
        path: Ordered list of TraversalStep.

    Returns:
        Numbered fragments with title and URL for citation.
    """
    blocks = []
    for index, step in enumerate(path, start=1):
        marker = " (forced jump — unrelated to the preceding fragment)" if step.forced_jump else ""
        blocks.append(
            f"[{index}] {step.article_title}{marker}\n"
            f"{step.article_url}\n\n"
            f"{step.text}"
        )
    return "\n\n---\n\n".join(blocks)


def material_path(output: str) -> str:
    """Derive the sidecar material path from the thesis output path.

    Sits beside the output rather than in the cwd, so the pair stays together
    and no file appears where it was not asked for.

    Args:
        output: The thesis output path.

    Returns:
        Path with ".material.md" replacing or appended to the output suffix.
    """
    path = Path(output)
    return str(path.with_suffix("")) + ".material.md"


def render_material(seed: str, path, config) -> str:
    """Render the walked fragments as a markdown audit trail.

    Records the knobs and every step's provenance so a thesis can be traced
    back to the exact material that produced it. The walk is non-deterministic,
    so without this the run is unreproducible once the process exits.

    Args:
        seed: The starting concept.
        path: Ordered list of TraversalStep.
        config: The TraversalConfig used for the walk.

    Returns:
        Markdown string describing the run and its fragments.
    """
    jumps = sum(1 for step in path if step.forced_jump)
    lines = [
        f"# Material: {seed}",
        "",
        f"Walked {len(path)} fragment(s) across "
        f"{len({step.article_title for step in path})} article(s), {jumps} forced jump(s).",
        "",
        f"Knobs: depth={config.depth}, epsilon={config.epsilon}, top_k={config.top_k}, "
        f"temperature={config.temperature}, "
        f"max_same_article_consecutive={config.max_same_article_consecutive}",
        "",
    ]
    for index, step in enumerate(path, start=1):
        marker = " — forced jump" if step.forced_jump else ""
        lines.append(
            f"## [{index}] {step.article_title}{marker}\n\n"
            f"*{step.article_url}* | similarity {step.similarity:.3f} | depth {step.depth}\n\n"
            f"{step.text}\n"
        )
    return "\n".join(lines)


def build_prompt(seed: str, path) -> str:
    """Assemble the synthesis prompt from a seed and a traversal path.

    Args:
        seed: The starting concept.
        path: Ordered list of TraversalStep.

    Returns:
        The prompt string.
    """
    return PROMPT.format(seed=seed, fragments=format_fragments(path))


def synthesize_idea(
    query: str,
    seed: str,
    traversal_config: TraversalConfig,
    embedder: Embedder,
    vector_store: VectorStoreClient,
    llm,
    llm_temperature: float,
) -> tuple[str, list, dict]:
    """Run a traversal then synthesize a thesis from its material.

    This is the core synthesis routine shared by the CLI command and the
    HTTP API. It performs no I/O beyond calling the supplied embedder,
    vector store, and LLM, so it can be invoked from any process.

    Args:
        query: Starting concept for the traversal (also used as the seed if
            the caller passed the same string for both).
        seed: Concept handed to the LLM as the synthesis lens.
        traversal_config: Resolved traversal knobs.
        embedder: Anything implementing the ``Embedder`` protocol.
        vector_store: A configured Qdrant-backed vector store.
        llm: Anything with a ``complete(messages, temperature) -> str``
            method (GatewayLLM or a stub).
        llm_temperature: Sampling temperature for the synthesis call.

    Returns:
        Tuple of (thesis, path, stats) where ``stats`` is a dict with
        ``depth``, ``epsilon``, ``top_k``, ``temperature``,
        ``max_same_article_consecutive``, ``forced_jumps``, and ``articles``.

    Raises:
        TraversalError: The walk could not be completed.
        EmbeddingError: The embedder rejected a request.
        GatewayError: The LLM call failed.
        ValueError: The walk produced an empty path.
    """
    path = run_traversal(query, traversal_config, embedder, vector_store)
    if not path:
        raise ValueError(
            "Walk produced no path. The corpus may be too small or the "
            "concept too specific."
        )

    thesis = synthesize_thesis(seed, path, llm, llm_temperature)
    stats = compute_stats(path, traversal_config)
    return thesis, path, stats


def run_traversal(query, traversal_config, embedder, vector_store) -> list:
    """Run a traversal and return its path. Pure pass-through to the engine.

    Args:
        query: Starting concept for the walk.
        traversal_config: Resolved traversal knobs.
        embedder: Anything implementing the ``Embedder`` protocol.
        vector_store: A configured Qdrant-backed vector store.

    Returns:
        Ordered list of TraversalStep.

    Raises:
        TraversalError: The walk could not be completed.
        EmbeddingError: The embedder rejected a request.
    """
    engine = TraversalEngine(
        embedder=embedder, vector_store=vector_store, config=traversal_config
    )
    return engine.traverse(query)


def synthesize_thesis(seed, path, llm, llm_temperature: float) -> str:
    """Hand a path to an LLM and return the synthesized thesis text.

    Args:
        seed: Concept handed to the LLM as the synthesis lens.
        path: Ordered list of TraversalStep (the walk material).
        llm: Anything with a ``complete(messages, temperature) -> str``
            method (GatewayLLM or a stub).
        llm_temperature: Sampling temperature for the synthesis call.

    Returns:
        Stripped thesis text.

    Raises:
        GatewayError: The LLM call failed.
    """
    return llm.complete(
        [{"role": "user", "content": build_prompt(seed, path)}],
        temperature=llm_temperature,
    ).strip()


def compute_stats(path, traversal_config) -> dict:
    """Summarize a traversal run as a small dict.

    Args:
        path: Ordered list of TraversalStep.
        traversal_config: Resolved traversal knobs.

    Returns:
        Dict with ``depth``, ``epsilon``, ``top_k``, ``temperature``,
        ``max_same_article_consecutive``, ``forced_jumps``, ``articles``.
    """
    return {
        "depth": traversal_config.depth,
        "epsilon": traversal_config.epsilon,
        "top_k": traversal_config.top_k,
        "temperature": traversal_config.temperature,
        "max_same_article_consecutive": traversal_config.max_same_article_consecutive,
        "forced_jumps": sum(1 for step in path if step.forced_jump),
        "articles": len({step.article_title for step in path}),
    }


@click.command()
@click.argument("concept")
@click.option("--depth", type=int, help="Maximum traversal depth (overrides config)")
@click.option("--epsilon", type=float, help="Exploration probability 0.0-1.0 (overrides config)")
@click.option("--top-k", type=int, help="Number of candidates per step (overrides config)")
@click.option(
    "--temperature",
    type=float,
    help="Softmax temperature for exploit path: 0=greedy, 1=natural, 2+=flat (overrides config)",
)
@click.option(
    "--max-same-article-consecutive",
    type=int,
    help="Hard block: max consecutive chunks from the same article (0=disabled, overrides config)",
)
@click.option("--model", type=str, help="LLM model name (overrides config)")
@click.option("--llm-temperature", type=float, help="Sampling temperature for the LLM")
@click.option("--output", "-o", type=click.Path(), help="Output file (default: stdout)")
@click.option(
    "--save-material/--no-save-material",
    default=False,
    help="Also write the walked fragments to '<output>.material.md' so the run is traceable. Requires -o.",
)
def idea(
    concept: str,
    depth: int | None,
    epsilon: float | None,
    top_k: int | None,
    temperature: float | None,
    max_same_article_consecutive: int | None,
    model: str | None,
    llm_temperature: float | None,
    output: str | None,
    save_material: bool,
):
    """Synthesize a thesis from a traversal of the corpus.

    Walks the vector space from CONCEPT exactly as `traverse` does, then hands
    the fragments to an LLM as material for a single argument. The traversal
    supplies material the model would not have reached for on its own; the
    model supplies the thinking that joins it.
    """
    cfg = get_config()

    if save_material and not output:
        click.echo(
            "Error: --save-material requires -o/--output, since the material is "
            "written alongside the thesis file.",
            err=True,
        )
        raise click.Abort()

    if not cfg.llm_gateway_url:
        click.echo(
            "Error: LLM_GATEWAY_URL is not set. `rhizome idea` needs an "
            "OpenAI-compatible endpoint serving chat completions and embeddings.",
            err=True,
        )
        raise click.Abort()

    depth = depth if depth is not None else cfg.default_depth
    epsilon = epsilon if epsilon is not None else cfg.epsilon
    top_k = top_k if top_k is not None else cfg.top_k
    temperature = temperature if temperature is not None else cfg.temperature
    max_same_article_consecutive = (
        max_same_article_consecutive
        if max_same_article_consecutive is not None
        else cfg.max_same_article_consecutive
    )
    llm_temperature = llm_temperature if llm_temperature is not None else cfg.llm_temperature

    click.echo(
        f"Walking: concept='{concept}', depth={depth}, epsilon={epsilon}, "
        f"temperature={temperature}",
        err=True,
    )

    embedder = GatewayEmbedder(
        base_url=cfg.llm_gateway_url,
        model=cfg.embedding_model,
        api_key=cfg.llm_gateway_api_key,
    )
    vector_store = VectorStoreClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection_name=cfg.qdrant_collection,
    )
    collection_mgr = CollectionManager(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)

    if not collection_mgr.collection_exists(cfg.qdrant_collection):
        click.echo(
            f"Collection '{cfg.qdrant_collection}' not found. Run `rhizome ingest` first.",
            err=True,
        )
        raise click.Abort()

    config = TraversalConfig(
        depth=depth,
        epsilon=epsilon,
        top_k=top_k,
        collection_name=cfg.qdrant_collection,
        temperature=temperature,
        max_same_article_consecutive=max_same_article_consecutive,
    )

    llm = GatewayLLM(
        base_url=cfg.llm_gateway_url,
        model=model or cfg.llm_model,
        api_key=cfg.llm_gateway_api_key,
    )

    try:
        path = run_traversal(concept, config, embedder, vector_store)
    except TraversalError as exc:
        click.echo(f"Traversal error: {exc}", err=True)
        raise click.Abort()
    except EmbeddingError as exc:
        click.echo(f"Embedding error: {exc}", err=True)
        raise click.Abort()

    if not path:
        click.echo(
            "No path generated. The corpus may be too small or the concept too specific.",
            err=True,
        )
        raise click.Abort()

    jumps = sum(1 for step in path if step.forced_jump)
    click.echo(
        f"Walked {len(path)} fragments across "
        f"{len({step.article_title for step in path})} articles ({jumps} forced jump(s))",
        err=True,
    )

    if save_material:
        if output is None:
            raise click.Abort()
        side = material_path(output)
        with open(side, "w", encoding="utf-8") as handle:
            handle.write(render_material(concept, path, config))
        click.echo(f"Material written to: {side}", err=True)

    click.echo(f"Synthesizing with {llm.model}…", err=True)

    try:
        thesis = synthesize_thesis(concept, path, llm, llm_temperature)
    except GatewayError as exc:
        click.echo(f"LLM error: {exc}", err=True)
        raise click.Abort()

    document = f"# {concept}\n\n{thesis}\n"

    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(document)
        click.echo(f"Output written to: {output}", err=True)
    else:
        click.echo(document)
