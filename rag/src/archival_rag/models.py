"""Shared data types for the RAG query path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ChainState(TypedDict, total=False):
    """LangGraph state for the RAG query graph."""

    question: str
    # context/language are per-turn: set from input (aquery/astream) or by the
    # enrich node, and reset to None by the generate node so the next turn
    # never reuses a stale retrieval.
    context: str | None
    language: str | None
    answer: str
    messages: Annotated[list[AnyMessage], add_messages]


@dataclass
class SourceDoc:
    source: str
    occupation: str
    year: int
    section: str
    chunk_type: str
    score: float  # RRF fusion score (rank-based), not cosine similarity


@dataclass
class RAGResult:
    answer: str
    sources: list[SourceDoc] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
