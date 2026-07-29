"""In-memory ingestion job tracking (adopted from Faiq's ingestion API draft).

Job state lives in process memory: restarts lose it and it is single-worker
only — acceptable for the lab service; swap for Redis/DB if that changes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class JobStatus(StrEnum):
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
    job = Job(id=job_id, filename=filename, status=JobStatus.PENDING)
    _jobs[job_id] = job
    return job


def update_job(job_id: str, **fields: Any) -> Job:
    job = _jobs[job_id].model_copy(update=fields)
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)
