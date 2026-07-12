import sys

from loguru import logger
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def setup_logging(level: str = "INFO") -> None:
    """Call once at process startup (main scripts and FastAPI lifespan)."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


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

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-m3"

    # --- RAG pipeline ---
    rag_top_k: int = 5
    # ef=256 at search time with m=32/ef_construct=200 index gives ~99%+ recall
    # and scales to millions of vectors. Switch exact=True only for debugging.
    rag_exact_search: bool = False
    rag_hnsw_ef: int = 256

    # --- Logging ---
    log_level: str = "INFO"
