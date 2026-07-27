"""Request/response models for the RAG service API."""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # One thread of conversation history per session_id (Drupal sends one
    # per chat widget instance).
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[\w.\-]+$")


class SourceRef(BaseModel):
    source: str
    occupation: str
    year: int
    section: str
    chunk_type: str
    score: float  # RRF fusion score (rank-based), not cosine similarity


class ChatResponse(BaseModel):
    answer: str
    declined: bool
    language: str | None = None
    session_id: str
    sources: list[SourceRef]
