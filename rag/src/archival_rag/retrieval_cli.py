#!/usr/bin/env python3
"""
retrieval_cli.py
----------------
Standalone CLI for the retrieval pipeline.

Workflow:
  1. User types a question.
  2. Recent conversation history is folded into the question (context-history),
     so follow-up questions like "what about page 2?" retrieve correctly.
  3. The question is embedded twice:
       - dense vector  -> via the VM's embedding service (HTTP call)
       - sparse vector -> via BM25, computed locally (fastembed)
  4. Both vectors are sent to Qdrant (on the VM, reached through the SSH
     tunnel) as one hybrid search request, fused with RRF.
  5. Top matching chunks are printed.

This script does NOT depend on the rest of the archival_rag package --
it only needs the tunnel to be running and the four settings below.
Run it from anywhere (repo root recommended) with:

    python retrieval_cli.py
    python retrieval_cli.py --query "your question"
    python retrieval_cli.py --session my_session
"""

import argparse
import json
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ============================================================
# CONFIG -- these must match your tunnel + Qdrant collection.
# Get these values from your .env / .env.vm if unsure.
# ============================================================

QDRANT_HOST = "localhost"
QDRANT_PORT = 16333  # tunneled port, not 6333
COLLECTION_NAME = "islandora_chunks"

EMBEDDING_API_URL = "http://localhost:18080"  # tunneled embedding service
EMBEDDING_API_KEY = "YOUR_API_KEY"  # from .env.vm

DENSE_VECTOR_NAME = "dense"  # must match collection schema
SPARSE_VECTOR_NAME = "bm25"  # must match collection schema
SPARSE_MODEL_NAME = "Qdrant/bm25"
BM25_LANGUAGE = "german"  # matches rag_bm25_language in project settings

TOP_K = 5
PREFETCH_LIMIT = 20  # candidates per branch before RRF fusion
HISTORY_WINDOW = 4  # number of past turns folded into query context

SESSIONS_DIR = Path("./sessions")
SESSIONS_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 1: embed the query (dense via VM service, sparse via local BM25)
# ============================================================


def embed_dense(text: str) -> list[float]:
    """Calls the embedding service running on the VM (through the tunnel)."""
    headers = {"X-API-Key": EMBEDDING_API_KEY} if EMBEDDING_API_KEY else {}
    response = httpx.post(
        f"{EMBEDDING_API_URL}/embed",
        json={"inputs": [text]},
        headers=headers,
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


_sparse_model = None  # loaded once, lazily


def embed_sparse(text: str) -> qmodels.SparseVector:
    """Computes a local BM25 sparse vector (no VM call needed for this part)."""
    global _sparse_model
    if _sparse_model is None:
        print("Loading local BM25 sparse model (first call only)...")
        _sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME, language=BM25_LANGUAGE
        )
    result = next(_sparse_model.embed([text]))
    return qmodels.SparseVector(
        indices=[int(i) for i in result.indices],
        values=[float(v) for v in result.values],
    )


# ============================================================
# STEP 2: context-history
# ============================================================


class ConversationHistory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.file_path = SESSIONS_DIR / f"{session_id}.jsonl"
        self.window = deque(maxlen=HISTORY_WINDOW)
        self._load_existing()

    def _load_existing(self):
        if self.file_path.exists():
            with open(self.file_path, encoding="utf-8") as f:
                for line in f:
                    self.window.append(json.loads(line))
            print(
                f"Resumed session '{self.session_id}' with"
                f" {len(self.window)} prior turn(s)."
            )

    def add_turn(self, query: str, answer_preview: str):
        turn = {
            "timestamp": datetime.now(UTC).isoformat(),
            "query": query,
            "answer_preview": answer_preview,
        }
        self.window.append(turn)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn) + "\n")

    def contextualize(self, new_query: str) -> str:
        if not self.window:
            return new_query
        lines = []
        for turn in self.window:
            lines.append(f"Q: {turn['query']}")
            if turn.get("answer_preview"):
                lines.append(f"A: {turn['answer_preview']}")
        return (
            "Conversation so far:\n"
            + "\n".join(lines)
            + f"\n\nCurrent question: {new_query}"
        )

    def show(self):
        if not self.window:
            print("(no history yet)")
            return
        for i, turn in enumerate(self.window, 1):
            print(f"[{i}] Q: {turn['query']}")

    def clear(self):
        self.window.clear()
        if self.file_path.exists():
            self.file_path.unlink()
        print("History cleared.")


# ============================================================
# STEP 3: hybrid search against Qdrant
# ============================================================


def hybrid_search(client: QdrantClient, query_text: str, top_k: int = TOP_K):
    dense_vec = embed_dense(query_text)
    sparse_vec = embed_sparse(query_text)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            qmodels.Prefetch(
                query=dense_vec, using=DENSE_VECTOR_NAME, limit=PREFETCH_LIMIT
            ),
            qmodels.Prefetch(
                query=sparse_vec, using=SPARSE_VECTOR_NAME, limit=PREFETCH_LIMIT
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return results.points


# ============================================================
# STEP 4: display results
# ============================================================


def format_hit(rank: int, point) -> str:
    payload = point.payload or {}
    text = payload.get("text", "<no text field found>")
    preview = (text[:300] + "...") if len(text) > 300 else text
    meta_keys = ["source", "doc_id", "page", "chunk_id", "section", "pid"]
    meta = (
        " | ".join(f"{k}={payload[k]}" for k in meta_keys if k in payload)
        or "no metadata fields matched"
    )
    return f"\n[{rank}] score={point.score:.4f}  ({meta})\n    {preview}"


def build_answer_preview(hits) -> str:
    if not hits:
        return ""
    return (hits[0].payload or {}).get("text", "")[:200]


# ============================================================
# MAIN WORKFLOW
# ============================================================


def run_single_query(client: QdrantClient, history: ConversationHistory, query: str):
    contextual_query = history.contextualize(query)
    hits = hybrid_search(client, contextual_query)

    if not hits:
        print("No results found.")
    else:
        print(f'\nTop {len(hits)} result(s) for: "{query}"')
        for i, hit in enumerate(hits, 1):
            print(format_hit(i, hit))

    history.add_turn(query, build_answer_preview(hits))


def repl(client: QdrantClient, history: ConversationHistory):
    print("\nArchive retrieval CLI. Type a question, or use /history, /reset, /exit.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not query:
            continue
        if query == "/exit":
            break
        if query == "/history":
            history.show()
            continue
        if query == "/reset":
            history.clear()
            continue
        run_single_query(client, history, query)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval CLI (hybrid search over Qdrant)"
    )
    parser.add_argument(
        "--query", type=str, help="Run a single query instead of entering the REPL"
    )
    parser.add_argument(
        "--session", type=str, default=None, help="Session id to persist/resume history"
    )
    args = parser.parse_args()

    session_id = args.session or f"session_{uuid.uuid4().hex[:8]}"
    print(f"Session: {session_id}  |  collection: {COLLECTION_NAME}")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    history = ConversationHistory(session_id)

    if args.query:
        run_single_query(client, history, args.query)
    else:
        repl(client, history)


if __name__ == "__main__":
    main()
