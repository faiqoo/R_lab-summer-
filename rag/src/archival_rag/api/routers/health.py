"""Health/readiness endpoints (unauthenticated)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from archival_rag.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    state = request.app.state
    points: int | None = None
    try:
        info = await state.qdrant.get_collection(state.settings.qdrant_collection)
        points = info.points_count
    except Exception:  # collection missing or Qdrant down — still report alive
        pass
    return HealthResponse(
        status="ok",
        collection=state.settings.qdrant_collection,
        points=points,
    )
