"""Upload staging for background ingestion (adopted from Faiq's draft).

Streams uploads to disk in 1MB chunks instead of reading them into memory.
The original filename is preserved inside a unique per-job directory: the
saved file's name becomes the chunk "source" payload (and feeds the
deterministic chunk IDs), so it must not be mangled with UUID prefixes.
Job directories are deleted after processing.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import UploadFile

_UPLOAD_BASE = Path(tempfile.gettempdir()) / "archival-rag-uploads"


async def save_upload(file: UploadFile) -> Path:
    """Stream one upload into a unique job directory, keeping its filename."""
    job_dir = _UPLOAD_BASE / str(uuid.uuid4())
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / Path(file.filename or "upload.pdf").name
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)
    return dest


def discard_upload(path: Path) -> None:
    """Remove a staged upload's job directory once it has been processed."""
    shutil.rmtree(path.parent, ignore_errors=True)
