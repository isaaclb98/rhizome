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

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rhizome.config import get_config
from rhizome.embedder.base import EmbeddingError
from rhizome.embedder.factory import get_embedder
from rhizome.traversal.config import TraversalConfig
from rhizome.traversal.engine import TraversalEngine, TraversalError
from rhizome.vectorstore.client import VectorStoreClient

log = logging.getLogger(__name__)

# ── Static file path ──────────────────────────────────────────────────────────

STATIC_DIR = Path(os.environ.get("RHIZOME_STATIC_DIR", "/app/static"))


# ── Request / Response models ─────────────────────────────────────────────────

class TraverseRequest(BaseModel):
    """POST /traverse request body."""

    query: str = Field(..., min_length=1, max_length=500)
    depth: int = Field(default=8, ge=1, le=20)
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=1, le=20)
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


class TraverseResponse(BaseModel):
    """POST /traverse response body."""

    path: list[TraversalStepResponse]
    stats: TraversalStatsResponse


# ── Application lifespan ───────────────────────────────────────────────────────

# Global instances (set during startup)
_embedder = None
_vector_store = None
_collection_manager = None
_config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize embedder and clients at startup."""
    global _embedder, _vector_store, _config

    _config = get_config()
    log.info("Initializing embedder: type=%s", _config.embedder_type)

    try:
        _embedder = get_embedder(
            embedder_type=_config.embedder_type,
            openai_api_key=_config.openai_api_key,
            hf_api_token=_config.hf_api_token,
            hf_model=_config.hf_model,
        )
    except Exception as e:
        log.error("Embedder initialization failed: %s", e)
        raise  # Fail fast — app cannot serve without embedder

    _vector_store = VectorStoreClient(
        url=_config.qdrant_url,
        api_key=_config.qdrant_api_key,
        collection_name=_config.qdrant_collection,
    )

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

# CORS for local development (Vite dev server on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check: verifies Qdrant connectivity.

    Returns 503 if Qdrant cannot be reached. The k8s readinessProbe
    uses this to determine when to route traffic to this pod.
    """
    try:
        _vector_store.client.get_collection(_config.qdrant_collection)
    except Exception as e:
        log.warning("Health check failed: collection '%s' unreachable: %s", _config.qdrant_collection, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant unavailable",
        )
    return {"status": "ok"}


@app.post("/traverse", response_model=TraverseResponse)
def traverse(req: TraverseRequest):
    """Run a rhizomatic traversal and return the path with metadata.

    The traversal walks through the vector space using epsilon-greedy search,
    returning each chunk with its domain, similarity score, and whether it
    was a forced global jump.
    """
    if _embedder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedder not initialized",
        )

    try:
        exists = _vector_store.client.collection_exists(_config.qdrant_collection)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant unavailable",
        )
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Collection '{_config.qdrant_collection}' not found",
        )

    config = TraversalConfig(
        depth=req.depth,
        epsilon=req.epsilon,
        top_k=req.top_k,
        collection_name=_config.qdrant_collection,
        temperature=req.temperature,
        max_same_article_consecutive=req.max_same_article_consecutive,
    )
    engine = TraversalEngine(
        embedder=_embedder,
        vector_store=_vector_store,
        config=config,
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
        ),
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
