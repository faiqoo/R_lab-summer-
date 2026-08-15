"""HTTP client for the embedding service (embedding_service.py on the GPU VM).

Mirrors the slice of the SentenceTransformer API this project uses, so
HybridSearcher and the ingest pipeline work unchanged against it.
"""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np


class EmbeddingServiceClient:
    # Server rejects requests above its input cap; stay under it per request.
    _MAX_INPUTS_PER_REQUEST = 256

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
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
