from pathlib import Path

import httpx
import pytest

from archival_rag.config import Settings
from scripts import snapshot


async def test_import_uploads_snapshot_as_multipart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Qdrant rejects a raw octet-stream body with 415; it needs multipart/form-data
    # with the file in a "snapshot" field.
    file = tmp_path / "chunks.snapshot"
    file.write_bytes(b"fake snapshot bytes")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen["content_type"] = request.headers["content-type"]
            seen["body"] = request.read().decode(errors="replace")
            return httpx.Response(200, json={"result": True})
        return httpx.Response(
            200, json={"result": {"points_count": 2, "status": "green"}}
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        snapshot.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )

    await snapshot.import_(
        Settings(qdrant_host="testserver", qdrant_port=6333, qdrant_collection="c"),
        file,
    )

    assert seen["content_type"].startswith("multipart/form-data")
    assert 'name="snapshot"' in seen["body"]
    assert "fake snapshot bytes" in seen["body"]
