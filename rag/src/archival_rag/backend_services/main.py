"""FastAPI application entrypoint. Run with: uvicorn archival_rag.api.main:app --reload"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import QdrantClient

from archival_rag.backend_services.routes.documents import router as documents_router
from archival_rag.backend_services.services.ingestion_service import ensure_qdrant_ready
from archival_rag.config import Settings
from archival_rag.retrieval.encoders import load_dense_encoder, load_sparse_encoder


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding models and connect to Qdrant once, reused across all requests."""
    settings = Settings()
    dense_model = load_dense_encoder(settings)
    sparse_model = load_sparse_encoder(settings)
    dim = dense_model.get_embedding_dimension()

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    ensure_qdrant_ready(client, settings, dim)

    app.state.settings = settings
    app.state.dense_model = dense_model
    app.state.sparse_model = sparse_model
    app.state.qdrant_client = client
    yield


app = FastAPI(title="Archival RAG Ingestion API", lifespan=lifespan)
app.include_router(documents_router)