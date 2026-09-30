"""FastAPI application for the Rhizome web visualizer.

Serves:
  GET  /health          — health check (verifies Qdrant connectivity)
  POST /traverse        — run a traversal and return path + stats
  POST /traverse/stream — streaming traversal (SSE)
  POST /idea            — run a traversal and synthesize a thesis
  POST /idea/stream     — streaming walk with thesis synthesis (SSE)
  GET  /{path:path}     — SPA fallback (serves index.html for non-API routes)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from rhizome.cli.commands.idea import (
    compute_stats,
    run_traversal,
    synthesize_thesis,
)
from rhizome.config import RhizomeConfig, get_config
from rhizome.embedder.base import Embedder, EmbeddingError
from rhizome.embedder.factory import get_embedder
from rhizome.gateway import GatewayEmbedder, GatewayError, GatewayLLM
from rhizome.traversal.config import TraversalConfig
from rhizome.traversal.engine import TraversalEngine, TraversalError
from rhizome.vectorstore.client import VectorStoreClient
from rhizome.vectorstore.collection import CollectionManager

log = logging.getLogger(__name__)

# ── Static file path ──────────────────────────────────────────────────────────

STATIC_DIR = Path(os.environ.get("RHIZOME_STATIC_DIR", "/app/static"))


# ── Dependency providers ──────────────────────────────────────────────────────
#
# These functions are the single source of truth for what each endpoint depends
# on. Tests use `app.dependency_overrides` to swap in fakes — production code
# calls the real factories via the lifespan.
# ─────────────────────────────────────────────────────────────────────────────


def get_embedder_dep() -> Embedder:
    """Return the shared embedder instance.

    Lifespan sets `app.state.embedder`. Tests override this dependency with a
    fake via `app.dependency_overrides[get_embedder_dep] = lambda: fake`.
    """
    raise RuntimeError(
        "get_embedder_dep called without lifespan initialization. "
        "Tests must use app.dependency_overrides[get_embedder_dep] = lambda: fake."
    )


def get_vector_store_dep() -> VectorStoreClient:
    """Return the shared vector store instance.

    Same override pattern as get_embedder_dep.
    """
    raise RuntimeError(
        "get_vector_store_dep called without lifespan initialization. "
        "Tests must use app.dependency_overrides[get_vector_store_dep] = lambda: fake."
    )


def get_config_dep() -> RhizomeConfig:
    """Return the cached RhizomeConfig singleton.

    Most tests can leave this on the default — only override when probing
    config-derived behavior (e.g., wikipedia_categories echoed in stats).
    """
    return get_config()


# ── Request / Response models ─────────────────────────────────────────────────

class TraverseRequest(BaseModel):
    """POST /traverse request body."""

    query: str = Field(..., min_length=1, max_length=500)
    depth: int = Field(default=8, ge=1, le=100)
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=1, le=50)
    temperature: float = Field(default=1.0, ge=0.0, le=3.0)
    max_same_article_consecutive: int = Field(default=2, ge=0, le=20)


class CandidateResponse(BaseModel):
    """A top_k candidate considered at a traversal step."""

    chunk_id: str
    text: str
    article_title: str
    article_url: str
    similarity: float


class TraversalStepResponse(BaseModel):
    """A single step in the traversal path response."""

    chunk_id: str
    text: str
    article_title: str
    article_url: str
    depth: int
    similarity: float
    forced_jump: bool
    candidates: list[CandidateResponse]


class TraversalStatsResponse(BaseModel):
    """Traversal statistics."""

    depth: int
    epsilon: float
    top_k: int
    forced_jumps: int
    temperature: float
    max_same_article_consecutive: int
    categories: str


class TraverseResponse(BaseModel):
    """POST /traverse response body."""

    path: list[TraversalStepResponse]
    stats: TraversalStatsResponse


class IdeaRequest(BaseModel):
    """POST /idea request body."""

    query: str = Field(..., min_length=1, max_length=500)
    seed: str | None = Field(default=None, max_length=500)
    depth: int = Field(default=8, ge=1, le=100)
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=1, le=50)
    temperature: float = Field(default=1.0, ge=0.0, le=3.0)
    max_same_article_consecutive: int = Field(default=2, ge=0, le=20)
    llm_model: str | None = Field(default=None, max_length=200)
    llm_temperature: float = Field(default=0.9, ge=0.0, le=3.0)


class IdeaStatsResponse(BaseModel):
    """Statistics returned with a synthesized thesis."""

    depth: int
    epsilon: float
    top_k: int
    temperature: float
    max_same_article_consecutive: int
    forced_jumps: int
    articles: int
    model: str


class IdeaResponse(BaseModel):
    """POST /idea response body."""

    thesis: str
    path: list[TraversalStepResponse]
    stats: IdeaStatsResponse


# ── Application lifespan ───────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize embedder and clients at startup.

    Production path: builds real embedder + vector store, attaches to app.state,
    and registers them as the default dependency implementations.

    Tests bypass this entirely via `app.dependency_overrides` and create the
    app with `TestClient(app)` *without* triggering lifespan (TestClient runs
    lifespan by default; pass `raise_server_exceptions=False` only if you want
    to inspect startup errors).
    """
    config = get_config()
    log.info("Initializing embedder: type=%s", config.embedder_type)

    try:
        embedder = get_embedder(
            embedder_type=config.embedder_type,
            openai_api_key=config.openai_api_key,
            hf_api_token=config.hf_api_token,
            hf_model=config.hf_model,
        )
    except Exception as e:
        log.error("Embedder initialization failed: %s", e)
        raise  # Fail fast — app cannot serve without embedder

    vector_store = VectorStoreClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=config.qdrant_collection,
    )

    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.config = config

    # Register production implementations. Tests override these.
    app.dependency_overrides[get_embedder_dep] = lambda: embedder
    app.dependency_overrides[get_vector_store_dep] = lambda: vector_store
    app.dependency_overrides[get_config_dep] = lambda: config

    log.info("Embedder and clients initialized successfully")
    yield
    # Cleanup on shutdown (nothing to clean up)


# ── App construction ───────────────────────────────────────────────────────────

app = FastAPI(
    title="Rhizome API",
    description="Rhizomatic traversal of Wikipedia embeddings",
    version="0.3.0",
    lifespan=lifespan,
)

# CORS — allowlist from env or default to localhost for dev
_allow_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in _allow_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health(
    vector_store: VectorStoreClient = Depends(get_vector_store_dep),
    config: RhizomeConfig = Depends(get_config_dep),
):
    """Health check: verifies Qdrant connectivity.

    Returns 503 if Qdrant cannot be reached. The k8s readinessProbe
    uses this to determine when to route traffic to this pod.
    """
    try:
        vector_store.client.get_collection(config.qdrant_collection)
    except Exception as e:
        log.warning("Health check failed: collection '%s' unreachable: %s", config.qdrant_collection, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant unavailable",
        )


@app.get("/config")
def config_endpoint(
    config: RhizomeConfig = Depends(get_config_dep),
):
    """Return the server configuration visible to clients."""
    return {
        "categories": config.wikipedia_categories,
    }


@app.post("/traverse", response_model=TraverseResponse)
def traverse(
    req: TraverseRequest,
    embedder: Embedder = Depends(get_embedder_dep),
    vector_store: VectorStoreClient = Depends(get_vector_store_dep),
    config: RhizomeConfig = Depends(get_config_dep),
):
    """Run a rhizomatic traversal and return the path with metadata.

    The traversal walks through the vector space using epsilon-greedy search,
    returning each chunk with its domain, similarity score, and whether it
    was a forced global jump.
    """
    try:
        exists = vector_store.client.collection_exists(config.qdrant_collection)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant unavailable",
        )
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Collection '{config.qdrant_collection}' not found",
        )

    traversal_config = TraversalConfig(
        depth=req.depth,
        epsilon=req.epsilon,
        top_k=req.top_k,
        collection_name=config.qdrant_collection,
        temperature=req.temperature,
        max_same_article_consecutive=req.max_same_article_consecutive,
    )
    engine = TraversalEngine(
        embedder=embedder,
        vector_store=vector_store,
        config=traversal_config,
    )

    try:
        path = engine.traverse(req.query)
    except TraversalError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Traversal failed: {e}",
        )
    except EmbeddingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding error: {e}",
        )

    forced_jumps = sum(1 for step in path if step.forced_jump)

    return TraverseResponse(
        path=[
            TraversalStepResponse(
                chunk_id=step.chunk_id,
                text=step.text,
                article_title=step.article_title,
                article_url=step.article_url,
                depth=step.depth,
                similarity=step.similarity,
                forced_jump=step.forced_jump,
                candidates=[
                    CandidateResponse(
                        chunk_id=c["payload"]["id"],
                        text=c["payload"]["text"],
                        article_title=c["payload"]["article_title"],
                        article_url=c["payload"]["article_url"],
                        similarity=float(c["score"]),
                    )
                    for c in step.candidates
                ],
            )
            for step in path
        ],
        stats=TraversalStatsResponse(
            depth=req.depth,
            epsilon=req.epsilon,
            top_k=req.top_k,
            forced_jumps=forced_jumps,
            temperature=req.temperature,
            max_same_article_consecutive=req.max_same_article_consecutive,
            categories=config.wikipedia_categories,
        ),
    )


@app.post("/traverse/stream", summary="Run a streaming traversal (SSE)")
async def traverse_stream(
    req: TraverseRequest,
    embedder: Embedder = Depends(get_embedder_dep),
    vector_store: VectorStoreClient = Depends(get_vector_store_dep),
    config: RhizomeConfig = Depends(get_config_dep),
):
    """Stream a traversal step-by-step using Server-Sent Events.

    Each event is a JSON line prefixed with 'data: '.
    Yields step-by-step as the traversal progresses — the frontend can render
    nodes incrementally as they arrive.
    """
    import asyncio
    import json

    try:
        vector_store.client.collection_exists(config.qdrant_collection)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant unavailable",
        )

    traversal_config = TraversalConfig(
        depth=req.depth,
        epsilon=req.epsilon,
        top_k=req.top_k,
        collection_name=config.qdrant_collection,
        temperature=req.temperature,
        max_same_article_consecutive=req.max_same_article_consecutive,
    )
    engine = TraversalEngine(
        embedder=embedder,
        vector_store=vector_store,
        config=traversal_config,
    )

    async def event_generator():
        forced_jumps = 0
        try:
            async for step in engine.traverse_stream(req.query):
                if step.forced_jump:
                    forced_jumps += 1
                yield f"data: {json.dumps({'type':'step','depth':step.depth,'chunk_id':step.chunk_id,'text':step.text,'article_title':step.article_title,'article_url':step.article_url,'similarity':step.similarity,'forced_jump':step.forced_jump,'candidates':[{'chunk_id':c['id'],'text':c['payload']['text'],'article_title':c['payload']['article_title'],'article_url':c['payload']['article_url'],'similarity':float(c['score'])} for c in step.candidates]})}\n\n"

            yield f"data: {json.dumps({'type':'done','path':engine.path,'stats':{'depth':req.depth,'epsilon':req.epsilon,'top_k':req.top_k,'temperature':req.temperature,'max_same_article_consecutive':req.max_same_article_consecutive,'forced_jumps':forced_jumps,'categories':config.wikipedia_categories}})}\n\n"
        except asyncio.CancelledError:
            # Yield a final done event so the client can clean up its streaming state.
            # engine.path contains whatever was accumulated before cancellation.
            yield f"data: {json.dumps({'type':'done','path':engine.path,'stats':{'depth':req.depth,'epsilon':req.epsilon,'top_k':req.top_k,'temperature':req.temperature,'max_same_article_consecutive':req.max_same_article_consecutive,'forced_jumps':forced_jumps,'categories':config.wikipedia_categories}})}\n\n"
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Static files + SPA fallback ───────────────────────────────────────────────

if STATIC_DIR.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def _path_to_response(path) -> list[TraversalStepResponse]:
    """Convert an internal path (list of TraversalStep) into API response items."""
    return [
        TraversalStepResponse(
            chunk_id=step.chunk_id,
            text=step.text,
            article_title=step.article_title,
            article_url=step.article_url,
            depth=step.depth,
            similarity=step.similarity,
            forced_jump=step.forced_jump,
            candidates=[
                CandidateResponse(
                    chunk_id=c["payload"]["id"],
                    text=c["payload"]["text"],
                    article_title=c["payload"]["article_title"],
                    article_url=c["payload"]["article_url"],
                    similarity=float(c["score"]),
                )
                for c in step.candidates
            ],
        )
        for step in path
    ]


def _require_collection(vector_store: VectorStoreClient, name: str) -> None:
    """Raise 503 if Qdrant is unreachable, 400 if the collection is missing."""
    try:
        exists = vector_store.client.collection_exists(name)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant unavailable",
        )
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Collection '{name}' not found",
        )


def _require_gateway(config: RhizomeConfig) -> str:
    """Return the configured LLM gateway URL, or raise 503 if it is unset."""
    if not config.llm_gateway_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "LLM gateway is not configured. Set LLM_GATEWAY_URL in the "
                "API container's environment."
            ),
        )
    return config.llm_gateway_url


def _get_llm_dep(
    config: RhizomeConfig = Depends(get_config_dep),
) -> GatewayLLM:
    """Build a GatewayLLM from config.

    Endpoints that want to honor a per-request ``llm_model`` override must
    call ``_build_llm(config, override)`` directly instead of using this
    dependency, since deps resolve before the request body is parsed.
    """
    return GatewayLLM(
        base_url=_require_gateway(config),
        model=config.llm_model,
        api_key=config.llm_gateway_api_key,
    )


def _build_llm(config: RhizomeConfig, model_override: str | None) -> GatewayLLM:
    return GatewayLLM(
        base_url=_require_gateway(config),
        model=model_override or config.llm_model,
        api_key=config.llm_gateway_api_key,
    )


@app.post("/idea", response_model=IdeaResponse)
def idea(
    req: IdeaRequest,
    embedder: Embedder = Depends(get_embedder_dep),
    vector_store: VectorStoreClient = Depends(get_vector_store_dep),
    config: RhizomeConfig = Depends(get_config_dep),
    llm: GatewayLLM = Depends(_get_llm_dep),
):
    """Run a traversal and synthesize a thesis from its material.

    Mirrors the ``rhizome idea`` CLI command but runs in-process inside the
    API container, so the visualizer's Synthesize tab can stream progress
    and final text over SSE without spawning a subprocess.
    """
    _require_collection(vector_store, config.qdrant_collection)
    _require_gateway(config)

    seed = req.seed or req.query
    traversal_config = TraversalConfig(
        depth=req.depth,
        epsilon=req.epsilon,
        top_k=req.top_k,
        collection_name=config.qdrant_collection,
        temperature=req.temperature,
        max_same_article_consecutive=req.max_same_article_consecutive,
    )

    try:
        path = run_traversal(req.query, traversal_config, embedder, vector_store)
    except TraversalError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Traversal failed: {exc}",
        )
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding error: {exc}",
        )

    if not path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Walk produced no path.",
        )

    # Honor per-request llm_model override if supplied; otherwise the
    # dependency-injected llm (built from config) is used.
    active_llm = _build_llm(config, req.llm_model) if req.llm_model else llm
    try:
        thesis = synthesize_thesis(seed, path, active_llm, req.llm_temperature)
    except GatewayError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed: {exc}",
        )

    stats = compute_stats(path, traversal_config)
    return IdeaResponse(
        thesis=thesis,
        path=_path_to_response(path),
        stats=IdeaStatsResponse(
            depth=stats["depth"],
            epsilon=stats["epsilon"],
            top_k=stats["top_k"],
            temperature=stats["temperature"],
            max_same_article_consecutive=stats["max_same_article_consecutive"],
            forced_jumps=stats["forced_jumps"],
            articles=stats["articles"],
            model=active_llm.model,
        ),
    )


@app.post("/idea/stream", summary="Stream a walk then synthesize a thesis (SSE)")
async def idea_stream(
    req: IdeaRequest,
    embedder: Embedder = Depends(get_embedder_dep),
    vector_store: VectorStoreClient = Depends(get_vector_store_dep),
    config: RhizomeConfig = Depends(get_config_dep),
    llm: GatewayLLM = Depends(_get_llm_dep),
):
    """Stream traversal steps as SSE events, then emit a final thesis event.

    Events:
      ``step``       — one per walked fragment, identical schema to /traverse/stream
      ``thesis``     — exactly one, fires after the LLM call completes
      ``done``       — exactly one, fires after ``thesis`` with the final stats

    The LLM call itself does not stream token-by-token; ``complete()`` blocks
    and yields one ``thesis`` event when it returns. Adding per-token streaming
    would require a streaming variant on GatewayLLM, which is out of scope.
    """
    import asyncio
    import json

    try:
        vector_store.client.collection_exists(config.qdrant_collection)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant unavailable",
        )

    seed = req.seed or req.query
    traversal_config = TraversalConfig(
        depth=req.depth,
        epsilon=req.epsilon,
        top_k=req.top_k,
        collection_name=config.qdrant_collection,
        temperature=req.temperature,
        max_same_article_consecutive=req.max_same_article_consecutive,
    )

    engine = TraversalEngine(
        embedder=embedder, vector_store=vector_store, config=traversal_config
    )

    async def event_generator():
        forced_jumps = 0
        path_holder: list = []
        try:
            async for step in engine.traverse_stream(req.query):
                if step.forced_jump:
                    forced_jumps += 1
                path_holder.append(step)
                yield f"data: {json.dumps({'type':'step','depth':step.depth,'chunk_id':step.chunk_id,'text':step.text,'article_title':step.article_title,'article_url':step.article_url,'similarity':step.similarity,'forced_jump':step.forced_jump,'candidates':[{'chunk_id':c['id'],'text':c['payload']['text'],'article_title':c['payload']['article_title'],'article_url':c['payload']['article_url'],'similarity':float(c['score'])} for c in step.candidates]})}\n\n"

            if not path_holder:
                yield f"data: {json.dumps({'type':'error','detail':'Walk produced no path.'})}\n\n"
                return

            try:
                thesis = await asyncio.to_thread(
                    synthesize_thesis,
                    seed,
                    path_holder,
                    llm,
                    req.llm_temperature,
                )
            except GatewayError as exc:
                yield f"data: {json.dumps({'type':'error','detail':f'LLM call failed: {exc}'})}\n\n"
                return

            stats = compute_stats(path_holder, traversal_config)
            yield f"data: {json.dumps({'type':'thesis','thesis':thesis})}\n\n"
            yield f"data: {json.dumps({'type':'done','stats':{**stats,'model':llm.model}})}\n\n"
        except asyncio.CancelledError:
            yield f"data: {json.dumps({'type':'cancelled'})}\n\n"
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/{path:path}")
async def spa_fallback(path: str):
    """Serve index.html for any non-API route to support client-side routing."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend not built. Run `cd rhizome/visualizer/app && npm install && npm run build`",
        )
    return FileResponse(str(index))
