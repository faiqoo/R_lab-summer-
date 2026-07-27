"""
RAG query graph: rewrite follow-ups → hybrid Qdrant search → prompt → ChatHAWKI.

Retrieval is hybrid (dense bge-m3 + BM25, RRF fusion — see retrieval/search.py).
Built as a LangGraph StateGraph with an in-memory checkpointer. Multi-turn
conversations are keyed by session_id (LangGraph thread_id): each session keeps
its own message history so follow-up questions have context. The session set
is a bounded LRU (rag_max_sessions) and each history is trimmed to
rag_history_max_messages — safe for a long-running service.

Related modules: prompts.py (all LLM prompts), models.py (state + result
types), formatting.py (language detection, result shaping).

Usage:
    from archival_rag.chain import build_chain
    from archival_rag.config import Settings

    rag = build_chain(Settings())

    # Single-turn (sync scripts/tests)
    result = rag.query("What does a baker apprenticeship cover?", session_id="s1")
    print(result.answer)
    for s in result.sources:
        print(s.source, s.score)

    # Async (FastAPI)
    result = await rag.aquery(
        "What does a baker apprenticeship cover?", session_id="s1"
    )

    # Token streaming (FastAPI SSE) — includes rewrite + score guard
    async for token in rag.astream("What about the exam?", session_id="s1"):
        print(token, end="", flush=True)

    # FastAPI lifespan shutdown
    await rag.aclose()
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger
from sentence_transformers import CrossEncoder

from archival_rag.config import Settings
from archival_rag.formatting import (
    DEFAULT_LANGUAGE,
    detect_language,
    extract_sources,
    format_docs,
)
from archival_rag.llm import ChatHAWKI
from archival_rag.models import ChainState, RAGResult, SourceDoc
from archival_rag.prompts import (
    FALLBACK_LOW_RELEVANCE,
    QUERY_REWRITE_PROMPT,
    RAG_ANSWER_PROMPT,
)
from archival_rag.retrieval.search import HybridSearcher


class RAGComponents:
    """Graph + hybrid retriever. FastAPI creates one instance at startup.

    aquery()/astream() are the managed entry points (LRU session tracking,
    score guard, trace). Invoking ``self.graph`` directly bypasses those.
    Call ``aclose()`` on shutdown to release Qdrant connections.
    """

    def __init__(
        self, settings: Settings, searcher: HybridSearcher | None = None
    ) -> None:
        self._settings = settings
        # Accept an externally shared searcher (the API lifespan passes one)
        # so encoders/clients exist once per process.
        self._searcher = searcher or HybridSearcher(settings)
        # LRU of session ids, oldest-touched first; bounded in _touch_session().
        self._sessions: OrderedDict[str, None] = OrderedDict()

        # NLI model for hallucination detection — only loaded when the feature is on.
        self._nli: CrossEncoder | None = None
        if settings.rag_hallucination_check:
            logger.info("Loading NLI model: {}", settings.rag_hallucination_model)
            self._nli = CrossEncoder(settings.rag_hallucination_model)

        llm = ChatHAWKI(
            api_url=settings.hawki_api_url,
            api_token=settings.hawki_token,  # type: ignore[arg-type]
            model=settings.hawki_model,
        )

        self._rewriter: Runnable[Any, Any] = (
            QUERY_REWRITE_PROMPT | llm | StrOutputParser()
        )
        rewriter = self._rewriter

        generation: Runnable[Any, Any] = RAG_ANSWER_PROMPT | llm | StrOutputParser()

        # aquery()/astream() retrieve before invoking the graph and pass
        # "context" and "language" in, so enrich only searches when the graph
        # is invoked directly without a precomputed context.
        def _enrich(state: ChainState) -> dict[str, Any]:
            update: dict[str, Any] = {}
            if state.get("context") is None:
                question = state["question"]
                history = self._trim(state.get("messages", []))
                if history:
                    question = rewriter.invoke(
                        {"question": question, "chat_history": history}
                    )
                _, results = self._searcher.search(question)
                update["context"] = format_docs(results)
            if not state.get("language"):
                update["language"] = detect_language(state["question"])
            return update

        async def _async_enrich(state: ChainState) -> dict[str, Any]:
            update: dict[str, Any] = {}
            if state.get("context") is None:
                question = state["question"]
                history = self._trim(state.get("messages", []))
                if history:
                    question = await rewriter.ainvoke(
                        {"question": question, "chat_history": history}
                    )
                _, results = await self._searcher.asearch(question)
                update["context"] = format_docs(results)
            if not state.get("language"):
                update["language"] = detect_language(state["question"])
            return update

        def _gen_inputs(state: ChainState) -> dict[str, Any]:
            return {
                "question": state["question"],
                "context": state.get("context") or "",
                "language": state.get("language") or DEFAULT_LANGUAGE,
                "chat_history": self._trim(state.get("messages", [])),
            }

        def _generate(state: ChainState) -> dict[str, Any]:
            answer = generation.invoke(_gen_inputs(state))
            return self._turn_update(state, answer)

        async def _async_generate(state: ChainState) -> dict[str, Any]:
            answer = await generation.ainvoke(_gen_inputs(state))
            return self._turn_update(state, answer)

        self._checkpointer = InMemorySaver()
        builder = StateGraph(ChainState)
        builder.add_node("enrich", RunnableLambda(func=_enrich, afunc=_async_enrich))
        builder.add_node(
            "generate", RunnableLambda(func=_generate, afunc=_async_generate)
        )
        builder.add_edge(START, "enrich")
        builder.add_edge("enrich", "generate")
        builder.add_edge("generate", END)
        # Callers pass: config={"configurable": {"thread_id": "<session-id>"}}
        self.graph = builder.compile(checkpointer=self._checkpointer)

    # ------------------------------------------------------------------
    # Session store (bounded LRU over checkpointer threads)
    # ------------------------------------------------------------------

    def _touch_session(self, session_id: str) -> None:
        self._sessions[session_id] = None
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._settings.rag_max_sessions:
            evicted_id, _ = self._sessions.popitem(last=False)
            self._checkpointer.delete_thread(evicted_id)
            logger.debug("Evicted LRU session: {}", evicted_id)

    def _trim(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        return messages[-self._settings.rag_history_max_messages :]

    def _turn_update(self, state: ChainState, answer: str) -> dict[str, Any]:
        """State update for one completed turn.

        Appends the human/AI pair, removes messages beyond the history cap so
        the persisted state stays bounded, and resets the per-turn keys.
        """
        messages = state.get("messages", [])
        overflow = len(messages) + 2 - self._settings.rag_history_max_messages
        removals = [
            RemoveMessage(id=m.id)
            for m in messages[: max(0, overflow)]
            if m.id is not None
        ]
        return {
            "answer": answer,
            "messages": [
                *removals,
                HumanMessage(content=state["question"]),
                AIMessage(content=answer),
            ],
            "context": None,
            "language": None,
        }

    # ------------------------------------------------------------------
    # Hallucination detection (NLI)
    # ------------------------------------------------------------------

    async def _async_hallucination_check(self, context: str, answer: str) -> bool:
        """Return True if answer appears unsupported by context.

        Uses an NLI cross-encoder. Label order for nli-deberta-v3-*:
        [contradiction=0, entailment=1, neutral=2].
        We flag when contradiction score is high (> 0.5), avoiding false
        positives on honest "I don't know" answers which score neutral.
        """
        if self._nli is None:
            return False
        # Truncate context to ~2000 chars so the NLI model's 512-token limit isn't hit.
        truncated_ctx = context[:2000]
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None, lambda: self._nli.predict([(truncated_ctx, answer)])  # type: ignore[union-attr]
        )
        contradiction_score = float(scores[0][0])
        hallucinated = contradiction_score > 0.5
        if hallucinated:
            logger.warning(
                "Hallucination detected (contradiction={:.2f}): {!r}",
                contradiction_score,
                answer[:120],
            )
        return hallucinated

    # ------------------------------------------------------------------
    # High-level query methods (score guard + trace + hallucination check)
    # ------------------------------------------------------------------

    async def _prepare(
        self, question: str, session_id: str
    ) -> tuple[RunnableConfig, dict[str, Any], list[Any]]:
        """Shared pre-generation steps: rewrite follow-ups, search once, guard.

        Returns (config, trace, results); results is empty when the query was
        declined (trace["declined"] is True). The guard runs on the top dense
        cosine score (trace["top_score"]), not the rank-based RRF scores.
        """
        cfg: RunnableConfig = {"configurable": {"thread_id": session_id}}
        trace: dict[str, Any] = {"session_id": session_id, "question": question}
        self._touch_session(session_id)

        snapshot = await self.graph.aget_state(cfg)
        history = self._trim(snapshot.values.get("messages", []))

        search_query = question
        if history:
            rewritten = await self._rewriter.ainvoke(
                {"question": question, "chat_history": history}
            )
            search_query = rewritten.strip() or question
            trace["search_query"] = search_query

        try:
            top_dense_score, results = await self._searcher.asearch(search_query)
        except Exception:
            logger.exception(
                "Qdrant search failed — session={} q={!r}", session_id, question[:80]
            )
            raise

        trace["top_score"] = round(top_dense_score, 4)
        trace["sources"] = [
            {"source": r.payload.get("source", ""), "score": round(r.score, 4)}
            for r in results
            if r.payload
        ]

        if top_dense_score < self._settings.rag_min_score:
            trace["declined"] = True
            trace["decline_reason"] = "low_relevance"
            logger.warning(
                "Query declined (top_score={:.3f} < {}) — session={} q={!r}",
                top_dense_score,
                self._settings.rag_min_score,
                session_id,
                question[:80],
            )
            return cfg, trace, []

        trace["declined"] = False
        return cfg, trace, results

    async def aquery(self, question: str, session_id: str) -> RAGResult:
        """Full traced async query: rewrite → retrieve once → score guard → generate.

        Retrieval happens exactly once: follow-up questions are rewritten into a
        self-contained query first, and the resulting documents are used for the
        score guard, the prompt context, the returned sources, and the
        hallucination check — so citations always match what the LLM saw.

        A detected hallucination is recorded in trace["hallucination_detected"]
        but the answer is returned unchanged; the caller decides how to react.
        """
        cfg, trace, results = await self._prepare(question, session_id)
        if trace["declined"]:
            return RAGResult(answer=FALLBACK_LOW_RELEVANCE, sources=[], trace=trace)

        language = detect_language(question)
        trace["language"] = language
        context = format_docs(results)

        try:
            state = await self.graph.ainvoke(
                {"question": question, "context": context, "language": language},
                config=cfg,
            )
        except Exception:
            logger.exception(
                "Generation failed — session={} q={!r}", session_id, question[:80]
            )
            raise
        answer = state["answer"]

        if self._settings.rag_hallucination_check:
            hallucinated = await self._async_hallucination_check(context, answer)
            trace["hallucination_detected"] = hallucinated
        else:
            trace["hallucination_detected"] = None

        trace["answer_length"] = len(answer)
        logger.info(
            "session={} top_score={:.3f} lang={} halluc={} q={!r}",
            session_id,
            trace["top_score"],
            language,
            trace["hallucination_detected"],
            question[:80],
        )

        sources = extract_sources(results)
        return RAGResult(answer=answer, sources=sources, trace=trace)

    async def astream(self, question: str, session_id: str) -> AsyncIterator[str]:
        """Stream answer tokens with the same rewrite + score guard as aquery.

        Declined queries yield the fallback message once. Models that don't
        support token streaming yield the full answer as a single chunk.
        No hallucination check — the answer is already on the wire.
        """
        cfg, trace, results = await self._prepare(question, session_id)
        if trace["declined"]:
            yield FALLBACK_LOW_RELEVANCE
            return

        language = detect_language(question)
        context = format_docs(results)

        emitted = False
        async for event in self.graph.astream(
            {"question": question, "context": context, "language": language},
            config=cfg,
            stream_mode="messages",
        ):
            chunk, metadata = cast("tuple[Any, dict[str, Any]]", event)
            # AIMessageChunk = LLM tokens; state writes (Human/AIMessage from
            # _turn_update) also flow through "messages" mode and must be skipped.
            if not isinstance(chunk, AIMessageChunk):
                continue
            content = chunk.content
            if metadata.get("langgraph_node") == "generate" and content:
                emitted = True
                yield str(content)
        if not emitted:
            snapshot = await self.graph.aget_state(cfg)
            yield snapshot.values.get("answer", "")

    def query(self, question: str, session_id: str) -> RAGResult:
        """Sync wrapper around aquery for scripts and tests.

        Must not be called from inside a running event loop — use aquery there.
        """
        return asyncio.run(self.aquery(question, session_id))

    def retrieve_sources(self, query: str) -> list[SourceDoc]:
        _, results = self._searcher.search(query)
        return extract_sources(results)

    async def async_retrieve_sources(self, query: str) -> list[SourceDoc]:
        _, results = await self._searcher.asearch(query)
        return extract_sources(results)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Release Qdrant connections. Call from FastAPI lifespan shutdown."""
        await self._searcher.aclose()


def build_chain(settings: Settings) -> RAGComponents:
    return RAGComponents(settings)
