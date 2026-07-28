# ragsvc — the RAG service the Drupal chatbot module calls (/chat, /chat/stream,
# /ingest, /health). See docs/deploying-ragsvc.md.
#
# Built directly on the target host (the GPU VM is x86_64 Rocky Linux; a Mac
# arm64 build would not run there without a separate multi-arch build step).
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first (cache layer): production deps only — no dev/eval
# groups, keeps deepeval/pandas/pytest out of the served image entirely.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

EXPOSE 8000

# --env-file .env at `podman run` time supplies HAWKI_TOKEN, QDRANT_HOST,
# EMBEDDING_API_URL, etc. — never baked into the image.
CMD ["uv", "run", "--no-sync", "uvicorn", "archival_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
