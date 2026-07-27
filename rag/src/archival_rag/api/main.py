"""FastAPI application factory for the RAG service (ragsvc).

This is the HTTP surface exposed to the Drupal chatbot module:
  - POST /ingest  — binary PDF upload → chunk → embed → upsert (routers/ingest.py)
  - GET  /health  — liveness + collection stats (routers/health.py)
  - chat endpoints follow in routers/chat.py (aquery/astream on RAGComponents)

Dense encoding follows the settings: with EMBEDDING_API_URL set it delegates
to the embedding service (on the VM: http://localhost:8080), otherwise it
loads the model in-process. Sparse (BM25) is always computed locally.

Run:
    uv run --env-file .env uvicorn archival_rag.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    Modifier,
    SparseVectorParams,
    VectorParams,
)

from archival_rag.api.routers import health, ingest
from archival_rag.config import Settings
from archival_rag.logging_setup import setup_logging
from archival_rag.retrieval.encoders import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    load_dense_encoder,
    load_sparse_encoder,
)


async def _ensure_collection(app: FastAPI) -> None:
    settings: Settings = app.state.settings
    if await app.state.qdrant.collection_exists(settings.qdrant_collection):
        return
    dim = app.state.dense.get_embedding_dimension()
    if dim is None:
        raise RuntimeError("Cannot determine embedding dimension")
    await app.state.qdrant.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
        },
        hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
    )
    logger.info("Created collection: {}", settings.qdrant_collection)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    setup_logging(settings.log_level)
    app.state.settings = settings
    app.state.dense = load_dense_encoder(settings)
    app.state.sparse = load_sparse_encoder(settings)
    app.state.qdrant = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=settings.qdrant_timeout,
    )
    app.state.api_key = (
        settings.embedding_api_key.get_secret_value()
        if settings.embedding_api_key
        else None
    )
    # Serialize CPU/GPU-heavy work: chunk+embed one request at a time.
    app.state.lock = asyncio.Lock()
    await _ensure_collection(app)
    yield
    await app.state.qdrant.close()


def create_app() -> FastAPI:
    app = FastAPI(title="archival-rag service", lifespan=_lifespan)
    app.include_router(health.router)
    app.include_router(ingest.router)
    return app


app = create_app()
