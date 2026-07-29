"""Chat endpoints — a thin HTTP layer over the RAG query graph (chain.py).

All RAG behavior (follow-up query rewriting, single hybrid retrieval,
relevance guard, per-session history with LRU eviction, optional
hallucination check) lives in RAGComponents. These endpoints only translate
HTTP to chain calls:

  - POST /chat         — full answer with sources
  - POST /chat/stream  — server-sent events; token frames, then a done frame
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from archival_rag.api.deps import require_api_key
from archival_rag.api.schemas import ChatRequest, ChatResponse, SourceRef

router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


def _rag(request: Request) -> Any:
    rag = request.app.state.rag
    if rag is None:
        raise HTTPException(
            status_code=503, detail="chat disabled: HAWKI_TOKEN not configured"
        )
    return rag


@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    rag = _rag(request)
    try:
        result = await rag.aquery(req.question, session_id=req.session_id)
    except Exception:
        logger.exception("chat failed — session={}", req.session_id)
        raise HTTPException(status_code=502, detail="generation failed") from None
    return ChatResponse(
        answer=result.answer,
        declined=bool(result.trace.get("declined", False)),
        language=result.trace.get("language"),
        session_id=req.session_id,
        sources=[SourceRef(**asdict(s)) for s in result.sources],
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """SSE stream: {"token": ...} frames, then {"done": true}.

    Declined (out-of-scope) questions stream the fallback message as a
    single token — same guard as /chat.
    """
    rag = _rag(request)

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for token in rag.astream(req.question, session_id=req.session_id):
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield 'data: {"done": true}\n\n'
        except Exception:
            logger.exception("chat stream failed — session={}", req.session_id)
            yield 'data: {"error": "generation failed"}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
