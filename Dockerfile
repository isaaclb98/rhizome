FROM python:3.14-slim AS builder

WORKDIR /app

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies into a virtual environment
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project --no-dev

# Final stage
FROM python:3.14-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source
COPY rhizome /app/rhizome
COPY config.yaml /app/config.yaml

# Install the package
RUN /app/.venv/bin/pip install --no-deps -e .

# Non-root user for security
RUN useradd --create-home rhizome
USER rhizome
WORKDIR /home/rhizome

ENV PATH="/home/rhizome/.local/bin:/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["rhizome"]
