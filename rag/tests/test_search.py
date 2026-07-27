"""Unit tests for HybridSearcher: request shape, guard unpacking, encoders."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client.models import FusionQuery, QueryRequest

from archival_rag.config import Settings
from archival_rag.retrieval import search as search_mod
from archival_rag.retrieval.search import HybridSearcher


class FakeDense:
    def encode(self, text: str) -> Any:
        return SimpleNamespace(tolist=lambda: [0.1, 0.2, 0.3])


class FakeSparse:
    def query_embed(self, text: str) -> Any:
        yield SimpleNamespace(indices=[7], values=[1.0])


class FakeSyncClient:
    def __init__(self) -> None:
        self.requests: list[list[QueryRequest]] = []
        self.guard_points = [SimpleNamespace(score=0.77, payload=None)]
        self.fused_points = [SimpleNamespace(score=0.5, payload={"source": "a.pdf"})]

    def query_batch_points(
        self, collection_name: str, requests: list[QueryRequest]
    ) -> list[Any]:
        self.requests.append(requests)
        return [
            SimpleNamespace(points=self.guard_points),
            SimpleNamespace(points=self.fused_points),
        ]

    def close(self) -> None:
        pass


class FakeAsyncClient(FakeSyncClient):
    async def query_batch_points(  # type: ignore[override]
        self, collection_name: str, requests: list[QueryRequest]
    ) -> list[Any]:
        return FakeSyncClient.query_batch_points(self, collection_name, requests)

    async def close(self) -> None:  # type: ignore[override]
        pass


@pytest.fixture
def searcher(monkeypatch: pytest.MonkeyPatch) -> HybridSearcher:
    monkeypatch.setattr(search_mod, "QdrantClient", lambda **kw: FakeSyncClient())
    monkeypatch.setattr(search_mod, "AsyncQdrantClient", lambda **kw: FakeAsyncClient())
    monkeypatch.setattr(search_mod, "load_dense_encoder", lambda s: FakeDense())
    monkeypatch.setattr(search_mod, "load_sparse_encoder", lambda s: FakeSparse())
    return HybridSearcher(Settings(hawki_token=None, _env_file=None))


def test_search_returns_guard_score_and_fused_points(
    searcher: HybridSearcher,
) -> None:
    top, points = searcher.search("Bäcker")
    assert top == 0.77
    assert points[0].payload["source"] == "a.pdf"


def test_batch_contains_guard_and_hybrid_requests(searcher: HybridSearcher) -> None:
    searcher.search("Bäcker")
    client: Any = searcher._client
    guard, hybrid = client.requests[0]

    # Guard: dense-only, top-1, no payload — feeds the rag_min_score check.
    assert guard.using == "dense"
    assert guard.limit == 1
    assert guard.with_payload is False

    # Hybrid: dense + sparse prefetch fused with RRF.
    assert isinstance(hybrid.query, FusionQuery)
    assert [p.using for p in hybrid.prefetch] == ["dense", "bm25"]
    assert all(p.limit == 20 for p in hybrid.prefetch)


def test_empty_guard_result_scores_zero(searcher: HybridSearcher) -> None:
    client: Any = searcher._client
    client.guard_points = []
    top, _ = searcher.search("völlig unbekannt")
    assert top == 0.0


async def test_asearch_matches_sync_results(searcher: HybridSearcher) -> None:
    top, points = await searcher.asearch("Bäcker")
    assert top == 0.77
    assert len(points) == 1


def test_shared_accessors_expose_internals(searcher: HybridSearcher) -> None:
    assert searcher.dense_encoder is searcher._dense
    assert searcher.sparse_encoder is searcher._sparse
    assert searcher.async_client is searcher._async_client
