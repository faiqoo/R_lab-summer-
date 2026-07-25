"""Tracks ingestion job status in memory (swap for Redis/DB if you need persistence)."""

from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Job(BaseModel):
    id: str
    filename: str
    status: JobStatus
    chunks_indexed: int = 0
    error: str | None = None


_jobs: dict[str, Job] = {}


def create_job(job_id: str, filename: str) -> Job:
    """Register a new job with PENDING status."""
    job = Job(id=job_id, filename=filename, status=JobStatus.PENDING)
    _jobs[job_id] = job
    return job


def update_job(job_id: str, **fields) -> Job:
    """Update one or more fields (status, chunks_indexed, error) of an existing job."""
    job = _jobs[job_id].model_copy(update=fields)
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Job | None:
    """Look up a job by id, returning None if it doesn't exist."""
    return _jobs.get(job_id)