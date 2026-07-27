"""Bridges the API layer to the existing ingestion pipeline (ingest.py) for one PDF at a time."""

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from archival_rag.config import Settings
from archival_rag.ingestion.ingest import (
    _ensure_collection,
    chunk_pdf,
    embed_and_build_points,
)


def ensure_qdrant_ready(client: QdrantClient, settings: Settings, dim: int) -> None:
    """Create the Qdrant collection on startup if it doesn't already exist."""
    _ensure_collection(client, settings.qdrant_collection, dim, recreate=False)


def ingest_pdf(
    path: Path,
    settings: Settings,
    dense_model: Any,
    sparse_model: Any,
    client: QdrantClient,
) -> int:
    """Chunk one PDF, embed its chunks, upsert into Qdrant, and return the chunk count."""
    chunks = chunk_pdf(path)
    if not chunks:
        return 0
    points = embed_and_build_points(chunks, dense_model, sparse_model)
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)