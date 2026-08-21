"""HTTP client for the embedding service (embedding_service.py on the GPU VM).

Mirrors the slice of the SentenceTransformer API this project uses, so
HybridSearcher and the ingest pipeline work unchanged against it.
"""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)


def _is_retryable(exc: BaseException) -> bool:
    """Retry 5xx responses and transient transport errors, never 4xx.

    Mirrors llm/client.py's HAWKI retry policy. Added after the SSH tunnel
    to the GPU VM dropped mid-request during a long eval run (2026-08-21) —
    encode() had no retry at all, unlike the HAWKI client, so a single
    transient disconnect crashed the whole multi-hour run instead of
    recovering.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


_embedding_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=1.0, max=20.0),
    stop=stop_after_attempt(4),
    reraise=True,
)


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

    @_embedding_retry
    def _post_embed(self, batch: list[str]) -> httpx.Response:
        response = self._client.post("/embed", json={"inputs": batch})
        response.raise_for_status()
        return response

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
            response = self._post_embed(batch)
            vectors_parts.append(
                np.asarray(response.json()["embeddings"], dtype=np.float32)
            )
        vectors = np.concatenate(vectors_parts)
        return vectors[0] if single else vectors

    @_embedding_retry
    def get_embedding_dimension(self) -> int | None:
        if self._dim is None:
            response = self._client.get("/health")
            response.raise_for_status()
            self._dim = int(response.json()["dim"])
        return self._dim
