FROM python:3.14-slim

WORKDIR /app

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy source and config
COPY pyproject.toml uv.lock* README.md rhizome /app/
COPY config.yaml /app/config.yaml

# Install dependencies and the project package
RUN uv sync --frozen --no-dev --no-install-project && \
    uv pip install pip hatchling editables && \
    uv pip install --no-deps -e . -v

# Non-root user for security
RUN useradd --create-home rhizome
USER rhizome
WORKDIR /home/rhizome

ENV PATH="/home/rhizome/.local/bin:/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["rhizome"]
