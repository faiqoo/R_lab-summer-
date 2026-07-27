"""Tests for the ragsvc API: multi-file PDF upload -> ingest."""

from __future__ import annotations

import io
from typing import Any

import fitz  # type: ignore[import-untyped]
import numpy as np
import pytest
from fastapi.testclient import TestClient

import archival_rag.api.main as app_mod
from archival_rag.api.main import app as fastapi_app


class FakeDense:
    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        return np.array([[0.1, 0.2, 0.3]] * len(texts), dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return 3


class FakeSparse:
    def passage_embed(self, texts: list[str]) -> Any:
        for _ in texts:
            yield type("E", (), {"indices": [1], "values": [1.0]})()


class FakeQdrant:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.upserts: list[Any] = []

    async def collection_exists(self, name: str) -> bool:
        return True

    async def get_collection(self, name: str) -> Any:
        return type("Info", (), {"points_count": len(self.upserts)})()

    async def upsert(self, collection_name: str, points: list[Any]) -> None:
        self.upserts.extend(points)

    async def close(self) -> None:
        pass


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeQdrant]:
    fake_qdrant = FakeQdrant()
    monkeypatch.setattr(app_mod, "load_dense_encoder", lambda s: FakeDense())
    monkeypatch.setattr(app_mod, "load_sparse_encoder", lambda s: FakeSparse())
    monkeypatch.setattr(app_mod, "AsyncQdrantClient", lambda **kw: fake_qdrant)
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    return TestClient(fastapi_app), fake_qdrant


def _pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def test_health_reports_collection(client: tuple[TestClient, FakeQdrant]) -> None:
    test_client, _ = client
    with test_client as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_uploads_and_upserts(client: tuple[TestClient, FakeQdrant]) -> None:
    test_client, fake_qdrant = client
    pdf = _pdf_bytes("Die Ausbildung zum Bäcker dauert drei Jahre.")

    with test_client as c:
        resp = c.post(
            "/ingest",
            files=[
                ("files", ("brd_baecker_1999.pdf", io.BytesIO(pdf), "application/pdf")),
                ("files", ("brd_metzger_2001.pdf", io.BytesIO(pdf), "application/pdf")),
            ],
        )

    assert resp.status_code == 200
    body = resp.json()
    assert [r["status"] for r in body["results"]] == ["ok", "ok"]
    assert body["total_chunks"] == len(fake_qdrant.upserts) > 0
    payloads = [p.payload for p in fake_qdrant.upserts]
    assert payloads[0]["source"] == "brd_baecker_1999.pdf"
    assert payloads[0]["occupation"] == "baecker"


def test_ingest_rejects_non_pdf(client: tuple[TestClient, FakeQdrant]) -> None:
    test_client, _ = client
    with test_client as c:
        resp = c.post(
            "/ingest",
            files=[("files", ("notes.txt", io.BytesIO(b"hallo"), "text/plain"))],
        )
    assert resp.status_code == 200
    assert resp.json()["results"][0]["status"] == "error"


def test_ingest_requires_api_key_when_configured(
    monkeypatch: pytest.MonkeyPatch, client: tuple[TestClient, FakeQdrant]
) -> None:
    test_client, _ = client
    monkeypatch.setenv("EMBEDDING_API_KEY", "geheim")
    pdf = _pdf_bytes("text")
    with test_client as c:
        denied = c.post(
            "/ingest", files=[("files", ("a.pdf", io.BytesIO(pdf), "application/pdf"))]
        )
        allowed = c.post(
            "/ingest",
            files=[("files", ("a.pdf", io.BytesIO(pdf), "application/pdf"))],
            headers={"X-API-Key": "geheim"},
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200


def test_ingest_caps_file_count(client: tuple[TestClient, FakeQdrant]) -> None:
    test_client, _ = client
    pdf = _pdf_bytes("x")
    files = [
        ("files", (f"f{i}.pdf", io.BytesIO(pdf), "application/pdf")) for i in range(51)
    ]
    with test_client as c:
        resp = c.post("/ingest", files=files)
    assert resp.status_code == 413


def test_async_ingest_creates_jobs_and_completes(
    client: tuple[TestClient, FakeQdrant],
) -> None:
    test_client, fake_qdrant = client
    pdf = _pdf_bytes("Die Ausbildung dauert drei Jahre.")

    with test_client as c:
        resp = c.post(
            "/ingest/async",
            files=[
                ("files", ("brd_koch_1998.pdf", io.BytesIO(pdf), "application/pdf"))
            ],
        )
        assert resp.status_code == 202
        job = resp.json()["jobs"][0]
        assert job["status"] == "pending"

        # TestClient runs background tasks before returning; job is finished.
        status = c.get(f"/status/{job['job_id']}").json()

    assert status["status"] == "done"
    assert status["chunks_indexed"] > 0
    assert len(fake_qdrant.upserts) == status["chunks_indexed"]


def test_async_ingest_marks_non_pdf_failed(
    client: tuple[TestClient, FakeQdrant],
) -> None:
    test_client, _ = client
    with test_client as c:
        resp = c.post(
            "/ingest/async",
            files=[("files", ("notes.txt", io.BytesIO(b"x"), "text/plain"))],
        )
        assert resp.status_code == 202
        job = resp.json()["jobs"][0]
        status = c.get(f"/status/{job['job_id']}").json()
    assert status["status"] == "failed"


def test_job_status_unknown_id_404(client: tuple[TestClient, FakeQdrant]) -> None:
    test_client, _ = client
    with test_client as c:
        resp = c.get("/status/nope")
    assert resp.status_code == 404
