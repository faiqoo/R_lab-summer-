"""Interactive retrieval REPL (based on Tahira Ghaffar's retrieval CLI).

Hybrid search (dense + BM25, RRF) against whatever your .env points at —
the VM stack through the tunnel, or a local setup. Built on the project's
HybridSearcher so it always matches the service's behavior (search params,
guard score) instead of drifting.

Sessions are logged as JSONL under rag/sessions/ (gitignored). History is
shown for orientation but deliberately NOT folded into the search query:
concatenating past turns dilutes both the dense vector and BM25 — follow-up
handling belongs to the chat pipeline's LLM query rewriting.

Usage:
    uv run --env-file .env python scripts/search_repl.py
    uv run --env-file .env python scripts/search_repl.py --query "Bäcker Ausbildung"
    uv run --env-file .env python scripts/search_repl.py --session demo
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from archival_rag.config import Settings
from archival_rag.retrieval.search import HybridSearcher

SESSIONS_DIR = Path(__file__).resolve().parents[1] / "sessions"


class SessionLog:
    """Append-only JSONL transcript of queries and top hits."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        SESSIONS_DIR.mkdir(exist_ok=True)
        self.file_path = SESSIONS_DIR / f"{session_id}.jsonl"
        self.turns: list[dict[str, Any]] = []
        if self.file_path.exists():
            with self.file_path.open(encoding="utf-8") as f:
                self.turns = [json.loads(line) for line in f]
            print(f"Resumed session '{session_id}' ({len(self.turns)} turns).")

    def add(self, query: str, top_source: str, top_score: float) -> None:
        turn = {
            "timestamp": datetime.now(UTC).isoformat(),
            "query": query,
            "top_source": top_source,
            "guard_score": round(top_score, 4),
        }
        self.turns.append(turn)
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    def show(self) -> None:
        if not self.turns:
            print("(no history yet)")
        for i, turn in enumerate(self.turns, 1):
            print(f"[{i}] {turn['query']}  -> {turn.get('top_source', '?')}")

    def clear(self) -> None:
        self.turns.clear()
        self.file_path.unlink(missing_ok=True)
        print("History cleared.")


def run_query(searcher: HybridSearcher, log: SessionLog, query: str) -> None:
    top_score, hits = searcher.search(query)
    print(f"\nguard score (dense cosine): {top_score:.3f}")
    if not hits:
        print("No results.")
        log.add(query, "", top_score)
        return
    for i, hit in enumerate(hits, 1):
        p = hit.payload or {}
        text = p.get("text", "")
        preview = text[:280] + ("..." if len(text) > 280 else "")
        occ = p.get("occupation") or "?"
        print(
            f"\n[{i}] rrf={hit.score:.4f}  {occ} | "
            f"{p.get('section', '?')} | {p.get('source', '?')}\n    {preview}"
        )
    top = hits[0].payload or {}
    log.add(query, top.get("source", ""), top_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval REPL")
    parser.add_argument("--query", help="run one query instead of the REPL")
    parser.add_argument("--session", default=None, help="session id to resume")
    args = parser.parse_args()

    settings = Settings()
    searcher = HybridSearcher(settings)
    session_id = args.session or f"session_{uuid.uuid4().hex[:8]}"
    log = SessionLog(session_id)
    print(f"Session: {session_id} | collection: {settings.qdrant_collection}")

    if args.query:
        run_query(searcher, log, args.query)
        return

    print("Type a question, or /history, /reset, /exit.\n")
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
            log.show()
            continue
        if query == "/reset":
            log.clear()
            continue
        run_query(searcher, log, query)


if __name__ == "__main__":
    main()
