"""Tests for the RAG graph: single retrieval, rewrite-first, LRU sessions, guard."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from archival_rag import chain as chain_mod
from archival_rag.chain import RAGComponents
from archival_rag.config import Settings
from archival_rag.formatting import detect_language
from archival_rag.prompts import FALLBACK_LOW_RELEVANCE
from archival_rag.retrieval import search as search_mod


def _point(score: float = 0.9, **overrides: Any) -> SimpleNamespace:
    payload = {
        "source": "baecker_1999.pdf",
        "occupation": "bäcker",
        "year": 1999,
        "section": "Ausbildungsdauer",
        "chunk_type": "text",
        "text": "Die Ausbildung dauert drei Jahre.",
    }
    payload.update(overrides)
    return SimpleNamespace(score=score, payload=payload)


def _messages(rag: RAGComponents, session_id: str) -> list[Any]:
    cfg = {"configurable": {"thread_id": session_id}}
    return rag.graph.get_state(cfg).values.get("messages", [])


class FakeEncoder:
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, text: str) -> Any:
        self.encoded.append(text)
        return SimpleNamespace(tolist=lambda: [0.1, 0.2, 0.3])


class FakeSparseEncoder:
    def query_embed(self, text: str) -> Any:
        yield SimpleNamespace(indices=[1, 2], values=[1.0, 1.0])


class FakeAsyncQdrant:
    """query_batch_points returns [guard response, fused response]."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[list[Any]] = []
        self.points: list[SimpleNamespace] = [_point()]
        self.closed = False

    async def query_batch_points(
        self, collection_name: str, requests: list[Any]
    ) -> list[SimpleNamespace]:
        self.calls.append(requests)
        return [
            SimpleNamespace(points=self.points[:1]),
            SimpleNamespace(points=self.points),
        ]

    async def close(self) -> None:
        self.closed = True


class FakeSyncQdrant:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[list[Any]] = []
        self.points: list[SimpleNamespace] = [_point()]
        self.closed = False

    def query_batch_points(
        self, collection_name: str, requests: list[Any]
    ) -> list[SimpleNamespace]:
        self.calls.append(requests)
        return [
            SimpleNamespace(points=self.points[:1]),
            SimpleNamespace(points=self.points),
        ]

    def close(self) -> None:
        self.closed = True


def make_components(
    monkeypatch: pytest.MonkeyPatch,
    llm_responses: list[str],
    **settings_overrides: Any,
) -> tuple[RAGComponents, FakeAsyncQdrant, FakeSyncQdrant, FakeEncoder]:
    async_client = FakeAsyncQdrant()
    sync_client = FakeSyncQdrant()
    encoder = FakeEncoder()

    monkeypatch.setattr(search_mod, "QdrantClient", lambda **kw: sync_client)
    monkeypatch.setattr(search_mod, "AsyncQdrantClient", lambda **kw: async_client)
    monkeypatch.setattr(search_mod, "load_dense_encoder", lambda s: encoder)
    monkeypatch.setattr(
        search_mod, "load_sparse_encoder", lambda s: FakeSparseEncoder()
    )
    monkeypatch.setattr(
        chain_mod, "ChatHAWKI", lambda **kw: FakeListChatModel(responses=llm_responses)
    )

    settings = Settings(hawki_token="test-token", _env_file=None, **settings_overrides)
    return RAGComponents(settings), async_client, sync_client, encoder


async def test_aquery_searches_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    rag, async_client, sync_client, _ = make_components(monkeypatch, ["answer one"])

    result = await rag.aquery(
        "What does a baker apprenticeship cover?", session_id="s1"
    )

    assert result.answer == "answer one"
    assert len(async_client.calls) == 1
    assert len(sync_client.calls) == 0


async def test_followup_searches_with_rewritten_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Response order: answer 1, rewrite for q2, answer 2.
    rag, async_client, _, encoder = make_components(
        monkeypatch, ["three years", "baker apprenticeship duration", "answer two"]
    )

    await rag.aquery("How long is a baker apprenticeship?", session_id="s1")
    result = await rag.aquery("What about the exam?", session_id="s1")

    assert len(async_client.calls) == 2
    assert encoder.encoded == [
        "How long is a baker apprenticeship?",
        "baker apprenticeship duration",
    ]
    assert result.trace["search_query"] == "baker apprenticeship duration"
    assert result.answer == "answer two"


async def test_low_score_declines_without_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rag, async_client, _, _ = make_components(monkeypatch, ["should never be used"])
    async_client.points = [_point(score=0.1)]

    result = await rag.aquery("What is the meaning of life?", session_id="s1")

    assert result.answer == FALLBACK_LOW_RELEVANCE
    assert result.sources == []
    assert result.trace["declined"] is True
    assert result.trace["decline_reason"] == "low_relevance"
    # Declined turn must not pollute the history.
    assert _messages(rag, "s1") == []


async def test_sources_match_retrieved_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rag, async_client, _, _ = make_components(monkeypatch, ["answer"])
    async_client.points = [
        _point(score=0.87),
        _point(score=0.61, source="metzger_2001.pdf"),
    ]

    result = await rag.aquery("How long is a baker apprenticeship?", session_id="s1")

    expected = ["baecker_1999.pdf", "metzger_2001.pdf"]
    assert [s.source for s in result.sources] == expected
    assert [s.score for s in result.sources] == [0.87, 0.61]
    assert result.trace["hallucination_detected"] is None


async def test_sessions_are_lru_evicted(monkeypatch: pytest.MonkeyPatch) -> None:
    rag, _, _, _ = make_components(monkeypatch, ["a1", "a2", "a3"], rag_max_sessions=2)

    await rag.aquery("Question one?", session_id="s1")
    await rag.aquery("Question two?", session_id="s2")
    await rag.aquery("Question three?", session_id="s3")

    assert list(rag._sessions) == ["s2", "s3"]
    # Eviction must also drop the checkpointed history.
    assert _messages(rag, "s1") == []
    assert len(_messages(rag, "s3")) == 2


async def test_history_is_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each turn: 1 rewrite (from turn 2 on) + 1 answer.
    responses = ["a1", "rw2", "a2", "rw3", "a3"]
    rag, _, _, _ = make_components(monkeypatch, responses, rag_history_max_messages=2)

    for q in ["Question one?", "Question two?", "Question three?"]:
        await rag.aquery(q, session_id="s1")

    # Persisted state stays bounded: only the last turn survives.
    messages = _messages(rag, "s1")
    assert len(messages) == 2
    assert messages[0].content == "Question three?"
    assert messages[1].content == "a3"


async def test_astream_yields_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    rag, _, _, _ = make_components(monkeypatch, ["streamed answer"])

    tokens = [
        t async for t in rag.astream("How long is the training?", session_id="s1")
    ]

    assert "".join(tokens) == "streamed answer"
    # The streamed turn is persisted like a normal turn.
    assert len(_messages(rag, "s1")) == 2


async def test_astream_declined_yields_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rag, async_client, _, _ = make_components(monkeypatch, ["unused"])
    async_client.points = [_point(score=0.1)]

    tokens = [t async for t in rag.astream("Meaning of life?", session_id="s1")]

    assert tokens == [FALLBACK_LOW_RELEVANCE]
    assert _messages(rag, "s1") == []


async def test_aclose_releases_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    rag, async_client, sync_client, _ = make_components(monkeypatch, ["a"])

    await rag.aclose()

    assert async_client.closed is True
    assert sync_client.closed is True


def testdetect_language_defaults_to_german() -> None:
    de = "Wie lange dauert die Bäckerausbildung in Deutschland?"
    en = "What does a baker apprenticeship cover in detail?"
    assert detect_language(de) == "German"
    assert detect_language(en) == "English"
    assert detect_language("12345 ???") == "German"
