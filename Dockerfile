# ── Stage 1: Build the React frontend ────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app

# Copy only what's needed for the frontend build
COPY rhizome/visualizer/app/package.json rhizome/visualizer/app/package-lock.json* ./

RUN npm ci --prefer-offline

COPY rhizome/visualizer/app/ ./
RUN npm run build

# ── Stage 2: FastAPI + static files ─────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python deps: FastAPI + Qdrant + both embedder SDKs
# The embedder selected at runtime by EMBEDDER_TYPE env var
RUN uv pip install --system fastapi uvicorn qdrant-client openai huggingface-hub

# Copy built frontend (from Stage 1)
COPY --from=frontend-builder /app/visualizer/static /app/static

# Set static files path for FastAPI SPA fallback
ENV RHIZOME_STATIC_DIR=/app/static

# Copy source
COPY pyproject.toml uv.lock* README.md .env.example /app/
COPY rhizome/ /app/rhizome/

# Install rhizome package (includes CLI, traversal, corpus, embedder)
RUN uv pip install --system /app

# Non-root user for security
RUN useradd --create-home rhizome
USER rhizome

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# API serves on port 8000; CMD runs the API server (not the CLI)
EXPOSE 8000
CMD ["uvicorn", "rhizome.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
