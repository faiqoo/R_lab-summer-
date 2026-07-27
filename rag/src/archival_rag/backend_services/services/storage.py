"""Saves incoming uploads to disk before they're handed to the ingestion pipeline."""

import uuid
from pathlib import Path

from fastapi import UploadFile

UPLOAD_DIR = Path("data/incoming")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def save_upload(file: UploadFile) -> Path:
    """Stream the upload to a unique per-job subfolder, keeping the original filename intact.

    The original filename matters downstream: chunk_pdf() uses the saved
    file's name as the "source" payload field in Qdrant, and _parse_filename()
    expects the "brd_<occupation>_<year>.pdf" convention at the start of the
    name — a UUID prefix on the filename itself would break both.
    """
    job_dir = UPLOAD_DIR / str(uuid.uuid4())
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / file.filename
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)
    return dest