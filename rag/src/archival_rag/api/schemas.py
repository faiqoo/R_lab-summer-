"""Request/response models for the RAG service API."""

from __future__ import annotations

from pydantic import BaseModel


class FileResult(BaseModel):
    file: str
    status: str  # "ok" | "empty" | "error"
    chunks: int = 0
    detail: str | None = None


class IngestResponse(BaseModel):
    results: list[FileResult]
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    collection: str
    points: int | None = None


class JobRef(BaseModel):
    job_id: str
    file: str
    status: str


class AsyncIngestResponse(BaseModel):
    jobs: list[JobRef]


class JobStatusResponse(BaseModel):
    id: str
    filename: str
    status: str
    chunks_indexed: int = 0
    error: str | None = None
