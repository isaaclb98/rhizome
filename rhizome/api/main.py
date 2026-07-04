"""FastAPI application for the Rhizome web visualizer.

Serves:
  GET  /health          — health check (verifies Qdrant connectivity)
  POST /traverse        — run a traversal and return path + stats
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

from rhizome.config import RhizomeConfig, get_config
from rhizome.embedder.base import Embedder, EmbeddingError
from rhizome.embedder.factory import get_embedder
from rhizome.traversal.config import TraversalConfig
from rhizome.traversal.engine import TraversalEngine, TraversalError
from rhizome.vectorstore.client import VectorStoreClient

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
