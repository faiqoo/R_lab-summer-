"""Embedding service — the C4 "Embedding Service" container, made real.

Hosts the dense encoder behind HTTP on the GPU VM so laptops and the RAG
service never load model weights. Clients use RemoteDenseEncoder
(retrieval/encoders.py) by setting EMBEDDING_API_URL.

Run on the VM:
    uv run uvicorn archival_rag.embedding_service:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from archival_rag.config import Settings
from archival_rag.logging_setup import setup_logging
from archival_rag.retrieval.encoders import load_local_dense_encoder

# Bounds one request to roughly one ingest batch — protects the GPU from a
# single caller monopolizing it (or OOMing it) with a giant payload.
_MAX_INPUTS = 256

# Max time a request waits to acquire the GPU lock before giving up. A
# 2026-08-02 incident: one slow/stuck encode() call left every request behind
# it in the queue waiting past the caller's own timeout (Drupal's cURL: 120s)
# with no useful error — just a silent hang. Failing fast here well under
# that gives callers an immediate, actionable 503 instead of a bare timeout,
# and stops an unbounded queue from building up behind one bad call.
_LOCK_WAIT_TIMEOUT = 90.0
# Threshold for logging a slow-encode warning — not a hard cutoff (the
# in-flight thread can't be safely cancelled once started; killing it here
# would let a second batch start concurrently and risk OOMing the GPU, the
# exact thing the lock exists to prevent). This just makes a repeat incident
# diagnosable from the dated log file instead of guesswork.
_SLOW_ENCODE_WARN_SECONDS = 30.0


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
    setup_logging(settings.log_level, service_name="embedsvc")
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

    try:
        await asyncio.wait_for(app.state.lock.acquire(), timeout=_LOCK_WAIT_TIMEOUT)
    except TimeoutError:
        logger.warning(
            "Rejecting /embed after waiting {}s for the GPU lock — "
            "another request is still holding it (batch_size={})",
            _LOCK_WAIT_TIMEOUT,
            len(req.inputs),
        )
        raise HTTPException(
            status_code=503,
            detail="embedding service busy processing another batch, retry shortly",
        ) from None

    try:
        started = time.monotonic()
        vectors = await loop.run_in_executor(
            None,
            lambda: app.state.encoder.encode(
                req.inputs, show_progress_bar=False, batch_size=16
            ),
        )
        elapsed = time.monotonic() - started
        if elapsed > _SLOW_ENCODE_WARN_SECONDS:
            logger.warning(
                "Slow /embed call: {:.1f}s for {} inputs (max input length {})",
                elapsed,
                len(req.inputs),
                max((len(t) for t in req.inputs), default=0),
            )
    finally:
        app.state.lock.release()

    return EmbedResponse(embeddings=[v.tolist() for v in vectors])
