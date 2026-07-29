"""Hybrid (dense + BM25) Qdrant search with RRF fusion.

Every search returns ``(top_dense_score, fused_points)``. The dense top score
exists because the low-relevance guard (rag_min_score) is calibrated on cosine
similarity, while RRF fusion scores are rank-based (~1/60) and carry no
absolute meaning. Both queries go to Qdrant in one batched round trip.
"""

from __future__ import annotations

import asyncio
from typing import Any

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    QueryRequest,
    SearchParams,
    SparseVector,
)

from archival_rag.config import Settings
from archival_rag.retrieval.encoders import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    load_dense_encoder,
    load_sparse_encoder,
    to_sparse_vector,
)


class HybridSearcher:
    """Encodes queries (dense + sparse) and runs fused search against Qdrant."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=settings.qdrant_timeout,
        )
        self._async_client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=settings.qdrant_timeout,
        )
        self._dense = load_dense_encoder(settings)
        self._sparse = load_sparse_encoder(settings)

    # Shared with the API layer so the process holds exactly one encoder set
    # and one client pair (ingest endpoints reuse them for embedding/upserts).
    @property
    def dense_encoder(self) -> Any:
        return self._dense

    @property
    def sparse_encoder(self) -> Any:
        return self._sparse

    @property
    def async_client(self) -> AsyncQdrantClient:
        return self._async_client

    def _encode(self, query: str) -> tuple[list[float], SparseVector]:
        dense = self._dense.encode(query).tolist()
        sparse = to_sparse_vector(next(iter(self._sparse.query_embed(query))))
        return dense, sparse

    def _requests(self, dense: list[float], sparse: SparseVector) -> list[QueryRequest]:
        s = self._settings
        params = SearchParams(hnsw_ef=s.rag_hnsw_ef, exact=s.rag_exact_search)
        return [
            # Guard query: top dense cosine score only, no payload.
            QueryRequest(
                query=dense,
                using=DENSE_VECTOR_NAME,
                params=params,
                limit=1,
                with_payload=False,
            ),
            # Hybrid query: dense + BM25 prefetch fused with reciprocal rank.
            QueryRequest(
                prefetch=[
                    Prefetch(
                        query=dense,
                        using=DENSE_VECTOR_NAME,
                        params=params,
                        limit=s.rag_prefetch_k,
                    ),
                    Prefetch(
                        query=sparse,
                        using=SPARSE_VECTOR_NAME,
                        limit=s.rag_prefetch_k,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=s.rag_top_k,
                with_payload=True,
            ),
        ]

    @staticmethod
    def _unpack(responses: list[Any]) -> tuple[float, list[Any]]:
        guard, fused = responses
        top_dense = guard.points[0].score if guard.points else 0.0
        return top_dense, fused.points

    def search(self, query: str) -> tuple[float, list[Any]]:
        dense, sparse = self._encode(query)
        responses = self._client.query_batch_points(
            collection_name=self._settings.qdrant_collection,
            requests=self._requests(dense, sparse),
        )
        return self._unpack(responses)

    async def asearch(self, query: str) -> tuple[float, list[Any]]:
        loop = asyncio.get_running_loop()
        dense, sparse = await loop.run_in_executor(None, self._encode, query)
        responses = await self._async_client.query_batch_points(
            collection_name=self._settings.qdrant_collection,
            requests=self._requests(dense, sparse),
        )
        return self._unpack(responses)

    async def aclose(self) -> None:
        await self._async_client.close()
        self._client.close()
