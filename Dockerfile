FROM python:3.14-slim

WORKDIR /app

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy source and config
COPY pyproject.toml uv.lock* README.md config.yaml /app/
COPY rhizome/ /app/rhizome/

# Install dependencies (the project itself is mounted/copied at runtime)
RUN uv sync --frozen --no-dev

# Non-root user for security
RUN useradd --create-home rhizome
USER rhizome
WORKDIR /home/rhizome

ENV PATH="/home/rhizome/.local/bin:/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "rhizome"]
