from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings for the Islandora Archival RAG system.

    All values can be overridden via environment variables or a .env file.
    See .env.example for the full list.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Auth ---
    # Optional during ingestion (not needed without the LLM query path)
    hawki_token: SecretStr | None = None

    # --- LLM (HAWKI) ---
    hawki_api_url: str = "https://chat.edu-ki-rlp.de"
    hawki_model: str = "qwen3-30b-a3b-instruct-2507"

    # --- Vector DB (Qdrant) ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "islandora_chunks"
    qdrant_timeout: int = 10

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-m3"
    # When set (e.g. http://141.26.157.121:8080), dense encoding is delegated
    # to the embedding service on the GPU VM instead of loading the model
    # locally. Unset = fully offline local encoding.
    embedding_api_url: str | None = None
    # Shared secret for the embedding service. Set the same value on the
    # server (checked on /embed) and on clients (sent as X-API-Key).
    embedding_api_key: SecretStr | None = None
    # httpx client timeout for EmbeddingServiceClient. 120s was too tight for
    # large PDFs during ingestion (2026-08 incident: documents ~5-9x the
    # typical size timed out — the encode itself was just slow, not stuck).
    # A single short chat query embeds in well under a second either way, so
    # raising the ceiling doesn't cost real-time query latency — it only
    # matters for the slow/large-batch case we want to tolerate better.
    embedding_timeout: float = 300.0

    # --- RAG pipeline ---
    rag_top_k: int = 5
    # Hybrid search: candidates fetched per branch (dense, BM25) before RRF fusion.
    rag_prefetch_k: int = 20
    rag_sparse_model: str = "Qdrant/bm25"
    rag_bm25_language: str = "german"
    # ef=256 at search time with m=32/ef_construct=200 index gives ~99%+ recall
    # and scales to millions of vectors. Switch exact=True only for debugging.
    rag_exact_search: bool = False
    rag_hnsw_ef: int = 256
    # Queries whose best chunk scores below this are declined (out-of-scope / noise).
    # Calibrated live against bge-m3 (2026-07-28): relevant queries score
    # 0.65–0.74 (cross-lingual included); out-of-scope ("Hauptstadt von
    # Frankreich") scored 0.544 and slipped past the old 0.4 threshold.
    rag_min_score: float = 0.6
    # NLI-based hallucination check — off by default (adds ~200 ms per query).
    rag_hallucination_check: bool = False
    rag_hallucination_model: str = "cross-encoder/nli-deberta-v3-small"

    # --- Sessions (in-memory chat history, single process) ---
    # LRU-evicted beyond this many concurrent sessions.
    rag_max_sessions: int = 1000
    # Per-session history is trimmed to the most recent N messages (human + AI).
    rag_history_max_messages: int = 20

    # --- Logging ---
    log_level: str = "INFO"

    # --- Evaluation (deepeval; see evaluation/ and docs/evaluation.md) ---
    # Deliberately NOT the same as hawki_model: grading with the same model
    # that generated the answer biases scores upward. openai-gpt-oss-120b is
    # HAWKI-hosted (no external OpenAI calls despite the name). Live-tested
    # 2026-07-28: reliable in isolation, but hit one JSON-parse failure across
    # ~15 real deepeval calls (more complex internal prompts than a simple
    # structured call) — evaluate()'s ErrorConfig already tolerates isolated
    # per-metric failures, so this is acceptable for judging, just not
    # perfectly clean. llama-3.3-70b-instruct was down (500s) when checked;
    # worth re-testing as an alternative once it's back.
    rag_eval_model: str = "openai-gpt-oss-120b"
    # Question-generation model for the synthesizer (dataset.py) — separate
    # from rag_eval_model because there's no self-grading bias concern here
    # (it isn't judging the chat pipeline's own output). qwen3-30b-a3b is the
    # most-tested model in this whole project (every live chat call this
    # session) and passed a live structured-output stress test 8/8 on
    # 2026-07-28, so it's the safer choice for a long unattended batch run.
    rag_synthesis_model: str = "qwen3-30b-a3b-instruct-2507"
    # HAWKI is a shared, courtesy-limited university endpoint, not a scaled
    # API. Governs both RAG-generation concurrency and deepeval's own
    # metric-judging concurrency in the harness. 4 balances a ~500-golden
    # target run finishing in reasonable time against not hammering a
    # shared endpoint — watch the first large run for errors/throttling
    # before raising further.
    rag_eval_max_concurrent: int = 4
    rag_eval_output_dir: str = "eval_runs"
