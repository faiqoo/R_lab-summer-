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
    hawki_token: SecretStr

    # --- LLM (HAWKI) ---
    hawki_api_url: str = "https://chat.edu-ki-rlp.de"
    hawki_model: str = "qwen3-30b-a3b-instruct-2507"

    # --- Vector DB (Qdrant) ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "islandora_chunks"

    # --- Embedding ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- RAG pipeline ---
    rag_top_k: int = 5
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
