"""Saves incoming uploads to disk before they're handed to the ingestion pipeline."""

import uuid
from pathlib import Path

from fastapi import UploadFile

UPLOAD_DIR = Path("data/incoming")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def save_upload(file: UploadFile) -> Path:
    """Stream the uploaded file to disk under a unique name and return its path."""
    dest = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)
    return dest