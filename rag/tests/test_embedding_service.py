"""Tests for the embedding service and its EmbeddingServiceClient client."""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from archival_rag import embedding_service
from archival_rag.config import Settings
from archival_rag.retrieval.encoders import EmbeddingServiceClient, load_dense_encoder


class FakeEncoder:
    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        return np.array([[0.1, 0.2, 0.3]] * len(texts), dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return 3


@pytest.fixture
def service_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        embedding_service, "load_local_dense_encoder", lambda s: FakeEncoder()
    )
    # Neutralize any developer .env value — env vars override the .env file,
    # and an empty key means auth is disabled.
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    return TestClient(embedding_service.app)


def test_health_reports_model_and_dim(service_client: TestClient) -> None:
    with service_client as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["dim"] == 3


def test_embed_returns_one_vector_per_input(service_client: TestClient) -> None:
    with service_client as client:
        resp = client.post("/embed", json={"inputs": ["eins", "zwei"]})
    assert resp.status_code == 200
    embeddings = resp.json()["embeddings"]
    assert len(embeddings) == 2
    assert embeddings[0] == pytest.approx([0.1, 0.2, 0.3])


def _mock_http_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"model": "test", "dim": 3})
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

    return httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handler)
    )


def test_remote_encoder_single_query() -> None:
    enc = EmbeddingServiceClient("http://testserver", client=_mock_http_client())

    vector = enc.encode("Wie lange dauert die Ausbildung?")

    assert vector.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert enc.get_embedding_dimension() == 3


def test_remote_encoder_batch() -> None:
    enc = EmbeddingServiceClient("http://testserver", client=_mock_http_client())

    vectors = enc.encode(["ein text"], show_progress_bar=False, batch_size=64)

    assert vectors.tolist() == [pytest.approx([0.1, 0.2, 0.3])]


def test_load_dense_encoder_dispatches_to_remote() -> None:
    settings = Settings(
        hawki_token="t", embedding_api_url="http://vm:8080", _env_file=None
    )
    assert isinstance(load_dense_encoder(settings), EmbeddingServiceClient)


def test_embed_rejects_oversized_batch(service_client: TestClient) -> None:
    with service_client as client:
        resp = client.post("/embed", json={"inputs": ["x"] * 257})
    assert resp.status_code == 422


def test_embed_requires_api_key_when_configured(
    monkeypatch: pytest.MonkeyPatch, service_client: TestClient
) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY", "geheim")
    with service_client as client:
        denied = client.post("/embed", json={"inputs": ["x"]})
        allowed = client.post(
            "/embed", json={"inputs": ["x"]}, headers={"X-API-Key": "geheim"}
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200
