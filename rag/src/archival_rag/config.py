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
    rag_min_score: float = 0.4
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
