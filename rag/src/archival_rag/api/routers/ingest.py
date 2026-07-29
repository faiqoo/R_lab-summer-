"""Document ingestion endpoints: binary PDF upload → chunk → embed → upsert.

Two modes:
  - POST /ingest        — synchronous; response carries per-file results
                          (used by the Node-RED indexing flow)
  - POST /ingest/async  — 202 + job ids; poll GET /status/{job_id}
                          (job pattern adopted from Faiq's ingestion API)
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    UploadFile,
)
from loguru import logger

from archival_rag.api import jobs
from archival_rag.api.deps import require_api_key
from archival_rag.api.schemas import (
    AsyncIngestResponse,
    FileResult,
    IngestResponse,
    JobRef,
    JobStatusResponse,
)
from archival_rag.api.storage import discard_upload, save_upload
from archival_rag.ingestion.ingest import chunk_pdf, embed_and_build_points

_MAX_FILES_PER_REQUEST = 50

router = APIRouter(tags=["ingest"], dependencies=[Depends(require_api_key)])


async def _process_pdf(state: Any, pdf_path: Path) -> int:
    """Chunk + embed one saved PDF and upsert it. Returns the chunk count."""
    loop = asyncio.get_running_loop()
    async with state.lock:
        chunks = await loop.run_in_executor(None, chunk_pdf, pdf_path)
        if not chunks:
            return 0
        points = await loop.run_in_executor(
            None, embed_and_build_points, chunks, state.dense, state.sparse
        )
    await state.qdrant.upsert(
        collection_name=state.settings.qdrant_collection, points=points
    )
    return len(points)


@router.post("/ingest")
async def ingest(files: list[UploadFile], request: Request) -> IngestResponse:
    """Upload PDFs (multipart) and index them into the vector collection.

    Re-uploading a file with the same name overwrites its previous chunks
    (deterministic IDs) instead of duplicating them.
    """
    if len(files) > _MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"too many files (max {_MAX_FILES_PER_REQUEST})",
        )

    state = request.app.state
    results: list[FileResult] = []
    total = 0

    for upload in files:
        name = upload.filename or "upload.pdf"
        if not name.lower().endswith(".pdf"):
            results.append(
                FileResult(file=name, status="error", detail="not a .pdf file")
            )
            continue
        pdf_path = await save_upload(upload)
        try:
            count = await _process_pdf(state, pdf_path)
            if count == 0:
                results.append(FileResult(file=name, status="empty"))
                continue
            total += count
            results.append(FileResult(file=name, status="ok", chunks=count))
            logger.info("Ingested {}: {} chunks", name, count)
        except Exception as exc:
            logger.exception("Ingest failed for {}", name)
            results.append(FileResult(file=name, status="error", detail=str(exc)))
        finally:
            discard_upload(pdf_path)

    return IngestResponse(results=results, total_chunks=total)


async def _run_background_jobs(state: Any, staged: list[tuple[str, Path]]) -> None:
    """Process staged uploads sequentially, recording per-job outcomes."""
    for job_id, pdf_path in staged:
        jobs.update_job(job_id, status=jobs.JobStatus.PROCESSING)
        try:
            count = await _process_pdf(state, pdf_path)
            jobs.update_job(job_id, status=jobs.JobStatus.DONE, chunks_indexed=count)
            logger.info("Job {}: {} chunks", job_id, count)
        except Exception as exc:
            logger.exception("Job {} failed", job_id)
            jobs.update_job(job_id, status=jobs.JobStatus.FAILED, error=str(exc))
        finally:
            discard_upload(pdf_path)


@router.post("/ingest/async", status_code=202)
async def ingest_async(
    files: list[UploadFile],
    request: Request,
    background_tasks: BackgroundTasks,
) -> AsyncIngestResponse:
    """Accept PDFs and queue them; poll GET /status/{job_id} for outcomes."""
    if len(files) > _MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"too many files (max {_MAX_FILES_PER_REQUEST})",
        )

    refs: list[JobRef] = []
    staged: list[tuple[str, Path]] = []
    for upload in files:
        name = upload.filename or "upload.pdf"
        job_id = str(uuid.uuid4())
        if not name.lower().endswith(".pdf"):
            jobs.create_job(job_id, name)
            jobs.update_job(
                job_id, status=jobs.JobStatus.FAILED, error="not a .pdf file"
            )
        else:
            jobs.create_job(job_id, name)
            staged.append((job_id, await save_upload(upload)))
        refs.append(
            JobRef(job_id=job_id, file=name, status=jobs.get_job(job_id).status)  # type: ignore[union-attr]
        )

    if staged:
        background_tasks.add_task(_run_background_jobs, request.app.state, staged)
    return AsyncIngestResponse(jobs=refs)


@router.get("/status/{job_id}")
async def job_status(job_id: str) -> JobStatusResponse:
    """Status of a previously submitted async ingestion job."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return JobStatusResponse(**job.model_dump())
