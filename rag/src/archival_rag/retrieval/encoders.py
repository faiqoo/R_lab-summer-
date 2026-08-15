"""Shared dense + sparse encoders and the Qdrant vector schema names.

Used by both the ingestion pipeline and query-time search so the two sides
can never drift apart on vector names or model choices.
"""

from __future__ import annotations

from typing import Any

import torch
from fastembed import SparseTextEmbedding
from loguru import logger
from qdrant_client.models import SparseVector
from sentence_transformers import SentenceTransformer

from archival_rag.config import Settings
from archival_rag.retrieval.embedding_client import EmbeddingServiceClient

# Named vectors in the Qdrant collection. Changing these requires re-ingesting.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DenseEncoder = SentenceTransformer | EmbeddingServiceClient


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
        return EmbeddingServiceClient(
            settings.embedding_api_url,
            api_key=api_key,
            timeout=settings.embedding_timeout,
        )
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
