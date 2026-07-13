"""Embedding service — the C4 "Embedding Service" container, made real.

Hosts the dense encoder behind HTTP on the GPU VM so laptops and the RAG
service never load model weights. Clients use RemoteDenseEncoder
(retrieval/encoders.py) by setting EMBEDDING_API_URL.

Run on the VM:
    uv run uvicorn archival_rag.embedding_service:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from archival_rag.config import Settings
from archival_rag.logging_setup import setup_logging
from archival_rag.retrieval.encoders import load_local_dense_encoder

# Bounds one request to roughly one ingest batch — protects the GPU from a
# single caller monopolizing it (or OOMing it) with a giant payload.
_MAX_INPUTS = 256


class EmbedRequest(BaseModel):
    inputs: list[str] = Field(min_length=1, max_length=_MAX_INPUTS)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


class HealthResponse(BaseModel):
    model: str
    dim: int


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    setup_logging(settings.log_level)
    app.state.model_name = settings.embedding_model
    # Loaded exactly once per process; every request reuses this instance.
    app.state.encoder = load_local_dense_encoder(settings)
    app.state.api_key = (
        settings.embedding_api_key.get_secret_value()
        if settings.embedding_api_key
        else None
    )
    # GPU encode calls are serialized: concurrent batches on a 10GB card OOM.
    app.state.lock = asyncio.Lock()
    yield


def _check_key(provided: str | None) -> None:
    expected = app.state.api_key
    if expected and provided != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


app = FastAPI(title="archival-rag embedding service", lifespan=_lifespan)


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(
        model=app.state.model_name,
        dim=app.state.encoder.get_embedding_dimension() or 0,
    )


@app.post("/embed")
async def embed(
    req: EmbedRequest,
    x_api_key: Annotated[str | None, Header()] = None,
) -> EmbedResponse:
    _check_key(x_api_key)
    loop = asyncio.get_running_loop()
    async with app.state.lock:
        vectors = await loop.run_in_executor(
            None,
            lambda: app.state.encoder.encode(
                req.inputs, show_progress_bar=False, batch_size=64
            ),
        )
    return EmbedResponse(embeddings=[v.tolist() for v in vectors])
