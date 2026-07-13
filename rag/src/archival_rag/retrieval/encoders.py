"""Shared dense + sparse encoders and the Qdrant vector schema names.

Used by both the ingestion pipeline and query-time search so the two sides
can never drift apart on vector names or model choices.
"""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np
import torch
from fastembed import SparseTextEmbedding
from loguru import logger
from qdrant_client.models import SparseVector
from sentence_transformers import SentenceTransformer

from archival_rag.config import Settings

# Named vectors in the Qdrant collection. Changing these requires re-ingesting.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RemoteDenseEncoder:
    """Client for the embedding service (embedding_service.py on the GPU VM).

    Mirrors the slice of the SentenceTransformer API this project uses, so
    HybridSearcher and the ingest pipeline work unchanged against it.
    """

    # Server rejects requests above its input cap; stay under it per request.
    _MAX_INPUTS_PER_REQUEST = 256

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=120.0, headers=headers
        )
        self._dim: int | None = None

    def encode(
        self,
        texts: str | list[str],
        show_progress_bar: bool = False,
        batch_size: int = 64,
    ) -> Any:
        single = isinstance(texts, str)
        inputs = [texts] if isinstance(texts, str) else list(texts)
        vectors_parts = []
        for start in range(0, len(inputs), self._MAX_INPUTS_PER_REQUEST):
            batch = inputs[start : start + self._MAX_INPUTS_PER_REQUEST]
            response = self._client.post("/embed", json={"inputs": batch})
            response.raise_for_status()
            vectors_parts.append(
                np.asarray(response.json()["embeddings"], dtype=np.float32)
            )
        vectors = np.concatenate(vectors_parts)
        return vectors[0] if single else vectors

    def get_embedding_dimension(self) -> int | None:
        if self._dim is None:
            response = self._client.get("/health")
            response.raise_for_status()
            self._dim = int(response.json()["dim"])
        return self._dim


DenseEncoder = SentenceTransformer | RemoteDenseEncoder


def load_local_dense_encoder(settings: Settings) -> SentenceTransformer:
    """Always loads model weights locally — used by the embedding service."""
    device = pick_device()
    encoder: SentenceTransformer = SentenceTransformer(
        settings.embedding_model, device=device
    )
    logger.info("Dense encoder: {}  device={}", settings.embedding_model, device)
    return encoder


def load_dense_encoder(settings: Settings) -> DenseEncoder:
    """Remote client when EMBEDDING_API_URL is set, local model otherwise."""
    if settings.embedding_api_url:
        logger.info("Dense encoder: remote via {}", settings.embedding_api_url)
        api_key = (
            settings.embedding_api_key.get_secret_value()
            if settings.embedding_api_key
            else None
        )
        return RemoteDenseEncoder(settings.embedding_api_url, api_key=api_key)
    return load_local_dense_encoder(settings)


def load_sparse_encoder(settings: Settings) -> SparseTextEmbedding:
    logger.info(
        "Sparse encoder: {}  language={}",
        settings.rag_sparse_model,
        settings.rag_bm25_language,
    )
    return SparseTextEmbedding(
        model_name=settings.rag_sparse_model,
        language=settings.rag_bm25_language,
    )


def to_sparse_vector(embedding: Any) -> SparseVector:
    return SparseVector(
        indices=[int(i) for i in embedding.indices],
        values=[float(v) for v in embedding.values],
    )
