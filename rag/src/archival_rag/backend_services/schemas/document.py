"""Pydantic models for the ingestion API's request/response bodies."""

from pydantic import BaseModel

from archival_rag.backend_services.services.jobs import JobStatus


class IngestResponse(BaseModel):
    job_id: str
    status: JobStatus
    filename: str


class JobStatusResponse(BaseModel):
    id: str
    filename: str
    status: JobStatus
    chunks_indexed: int = 0
    error: str | None = None