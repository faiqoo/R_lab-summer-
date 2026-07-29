"""Helpers shared by the RAG query path: language detection, result shaping."""

from __future__ import annotations

from typing import Any

from langdetect import LangDetectException  # type: ignore[import-untyped]
from langdetect import detect as _langdetect

from archival_rag.models import SourceDoc

_LANG_NAMES = {"en": "English", "de": "German"}

# The corpus and most users are German; langdetect is unreliable on short
# queries, so German is the safer fallback than English.
DEFAULT_LANGUAGE = "German"


def detect_language(text: str) -> str:
    try:
        code = _langdetect(text)
        return _LANG_NAMES.get(code, DEFAULT_LANGUAGE)
    except LangDetectException:
        return DEFAULT_LANGUAGE


def extract_sources(results: list[Any]) -> list[SourceDoc]:
    """Map Qdrant scored points to SourceDoc entries."""
    docs = []
    for r in results:
        p = r.payload or {}
        docs.append(
            SourceDoc(
                source=p.get("source", ""),
                occupation=p.get("occupation", ""),
                year=p.get("year", 0),
                section=p.get("section", ""),
                chunk_type=p.get("chunk_type", "text"),
                score=round(r.score, 4),
            )
        )
    return docs


def format_docs(results: list[Any]) -> str:
    """Render retrieved chunks as the Documents block of the answer prompt."""
    chunks = []
    for r in results:
        p = r.payload or {}
        parts = []
        if p.get("occupation"):
            parts.append(p["occupation"].title())
        if p.get("year"):
            parts.append(str(p["year"]))
        if p.get("section"):
            parts.append(p["section"])
        if p.get("source"):
            parts.append(f"({p['source']})")
        header = " — ".join(parts)
        chunks.append(f"[{header}]\n{p.get('text', '')}")
    return "\n\n---\n\n".join(chunks)
