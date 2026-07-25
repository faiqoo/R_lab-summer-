"""Endpoints the 3rd party (e.g. Drupal) calls to hand us a document and check on it."""

import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile

from archival_rag.backend_services.schemas.document import IngestResponse, JobStatusResponse
from archival_rag.backend_services.services.ingestion_service import ingest_pdf
from archival_rag.backend_services.services.jobs import JobStatus, create_job, get_job, update_job
from archival_rag.backend_services.services.storage import save_upload

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def _run_ingestion(job_id: str, path, settings, dense_model, sparse_model, client) -> None:
    """Background worker: ingest one file into Qdrant and record the outcome."""
    try:
        update_job(job_id, status=JobStatus.PROCESSING)
        count = ingest_pdf(path, settings, dense_model, sparse_model, client)
        update_job(job_id, status=JobStatus.DONE, chunks_indexed=count)
    except Exception as exc:
        update_job(job_id, status=JobStatus.FAILED, error=str(exc))


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> IngestResponse:
    """Accept a PDF pushed by the 3rd party and queue it for background ingestion."""
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only application/pdf is accepted")

    path = await save_upload(file)
    job_id = str(uuid.uuid4())
    create_job(job_id, file.filename)

    state = request.app.state
    background_tasks.add_task(
        _run_ingestion,
        job_id,
        path,
        state.settings,
        state.dense_model,
        state.sparse_model,
        state.qdrant_client,
    )
    return IngestResponse(job_id=job_id, status=JobStatus.PENDING, filename=file.filename)


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_ingestion_status(job_id: str) -> JobStatusResponse:
    """Return the current status of a previously submitted ingestion job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    return JobStatusResponse(**job.model_dump())