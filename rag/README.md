# archival-rag

Python package for the **Islandora Archival Chatbot** (Summer 2026 lab, Uni Koblenz, IST). Builds a full RAG pipeline on top of the Berufearchiv digital archive — letting researchers ask natural-language questions over ~5,400 digitised vocational training documents.

## First-Time Setup

```bash
cp .env.example .env        # fill in HAWKI_TOKEN at minimum
make install                # uv sync + pre-commit hooks
```

Get `HAWKI_TOKEN` from [chat.edu-ki-rlp.de](https://chat.edu-ki-rlp.de) → Profile → API Tokens.

---

## Ingestion

Chunk PDFs, embed with `BAAI/bge-m3`, and upsert into Qdrant.

```bash
make ingest-recreate                          # full ingest (drops + rebuilds collection)
make ingest                                   # append-only (skip --recreate)
make ingest-recreate PDF_DIR=/custom/path     # custom PDF directory
```

Qdrant + the embedding service run on the shared GPU VM — see [docs/gpu-vm-services.md](docs/gpu-vm-services.md) to connect via the tunnel (`make tunnel`). Working fully offline instead? Run a local Qdrant (`docker run -p 6333:6333 qdrant/qdrant`) and remove the VM `.env` lines — same doc covers the fallback. The first run downloads the bge-m3 model (~2 GB) if no `EMBEDDING_API_URL` is set. Ingesting all 5,400 PDFs takes ~1–2 hours.

### Snapshot (share with colleagues)

Instead of re-ingesting, share a snapshot of the Qdrant collection:

```bash
# You — export after ingestion:
make export                                         # → islandora_chunks.snapshot
make export SNAPSHOT_FILE=my_snapshot.snapshot      # custom filename

# Colleagues — import on their machine:
make import-snapshot SNAPSHOT_FILE=islandora_chunks.snapshot
```

---

## RAG Chain

`build_chain()` returns a `RAGComponents` instance — use `aquery`/`astream` (async, preferred) or `query` (sync wrapper, for one-off scripts):

```python
import asyncio
from archival_rag.chain import build_chain
from archival_rag.config import Settings

async def main() -> None:
    rag = build_chain(Settings())

    # Single question — result.answer / result.sources / result.trace
    result = await rag.aquery("What does a baker apprenticeship cover?", session_id="thread-1")
    print(result.answer)

    # Follow-up in the same thread — history is remembered, query is
    # rewritten to resolve pronouns like "it"/"this" before retrieval
    result = await rag.aquery("How long does it last?", session_id="thread-1")
    print(result.answer)

    # Get source documents without calling the LLM
    for s in await rag.async_retrieve_sources("baker apprenticeship duration"):
        print(s.source, s.section, s.score)

    await rag.aclose()  # release Qdrant connections

asyncio.run(main())
```

Each `session_id` is an independent conversation thread with its own history. The chain answers in the same language as the question (German or English). A sync `rag.query(question, session_id)` wrapper exists for quick scripts/REPLs — don't call it from inside a running event loop, use `aquery` there instead.

### Streaming (FastAPI SSE)

```python
async for token in rag.astream("What does a baker apprenticeship cover?", session_id="thread-1"):
    print(token, end="", flush=True)
```

---

## Running the API (ragsvc)

The HTTP service the Drupal chatbot module calls (`POST /chat`, `POST /ingest`, `GET /health`) lives at `archival_rag.api` — a package (`src/archival_rag/api/`), not a single file.

```bash
make api    # uvicorn on :8000, local dev/testing only
```

**This is not how the Drupal team reaches the service.** In practice `ragsvc` runs as a container on the GPU VM (podman + systemd), alongside Qdrant and the embedding service — see [docs/running-the-full-stack.md](docs/running-the-full-stack.md) for the real topology and [docs/deploying-ragsvc.md](docs/deploying-ragsvc.md) for deploying code changes there. `make api` is just for running the chain locally against your own `.env`/tunnel while developing.

> Seeing a stale `backend_services/` directory with only `.pyc` files and no source? That app was folded into `api/` in `f4ace03` (refactor: consolidate ingestion APIs). The `.py` files are gone on `feature/rag`; `__pycache__/` isn't git-tracked so a pull won't clean it up — safe to delete: `find src -name __pycache__ -type d -exec rm -rf {} +`.

---

## Direct LLM Usage

```python
from archival_rag.llm.chat_model import ChatHAWKI
from archival_rag.config import Settings

settings = Settings()
llm = ChatHAWKI(
    api_url=settings.hawki_api_url,
    api_token=settings.hawki_token,
    model=settings.hawki_model,
)
response = llm.invoke("What is archival science?")
print(response.content)
```

---

## Available Models

| Model | Identifier |
| ----- | ---------- |
| Qwen 3 30B *(default)* | `qwen3-30b-a3b-instruct-2507` |
| Llama 3.1 8B | `meta-llama-3.1-8b-instruct` |
| Llama 3.3 70B | `llama-3.3-70b-instruct` |
| Qwen 3 Coder 30B | `qwen3-coder-30b-a3b-instruct` |
| Qwen 3 Omni 30B | `qwen3-omni-30b-a3b-instruct` |
| Gemma 4 31B | `gemma-4-31b-it` |
| Devstral 123B | `devstral-2-123b-instruct-2512` |
| Apertus 70B | `apertus-70b-instruct-2509` |
| OSS 120B | `openai-gpt-oss-120b` |

Use `HAWKIModel` enum for type safety: `from archival_rag.llm.chat_model import HAWKIModel`.

---

## Environment Variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `HAWKI_TOKEN` | *(required for queries)* | Bearer token from HAWKI web UI |
| `HAWKI_API_URL` | `https://chat.edu-ki-rlp.de` | HAWKI base URL |
| `HAWKI_MODEL` | `qwen3-30b-a3b-instruct-2507` | LLM model |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant REST port |
| `QDRANT_COLLECTION` | `islandora_chunks` | Collection name |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model (8192-token context, multilingual) |
| `RAG_TOP_K` | `5` | Chunks retrieved per query |
| `RAG_EXACT_SEARCH` | `false` | Use exact search instead of HNSW (debug only) |
| `RAG_HNSW_EF` | `256` | HNSW search beam width (~99% recall at scale) |

---

## Dev

```bash
make test       # pytest
make lint       # mypy + ruff + black --check
make format     # black + ruff --fix
make check      # lint + test
```

## Evaluation

Retrieval and generation quality are measured with a `deepeval`-based harness (golden dataset + RAG metrics, judged via HAWKI). Heavy/eval-only deps are a separate install group:

```bash
make install-eval
make eval ARGS="--limit 5"    # cheap smoke run
```

See [docs/evaluation.md](docs/evaluation.md) for the full workflow (dataset synthesis, metrics, comparing runs).

## Further docs

| Doc | Covers |
| --- | --- |
| [docs/gpu-vm-services.md](docs/gpu-vm-services.md) | Connecting to the shared GPU VM's Qdrant + embedding service |
| [docs/running-the-full-stack.md](docs/running-the-full-stack.md) | How every piece (Drupal, VM services, ragsvc) fits together and starts up |
| [docs/deploying-ragsvc.md](docs/deploying-ragsvc.md) | Deploying `ragsvc` (the API the Drupal team calls) on the VM |
| [docs/evaluation.md](docs/evaluation.md) | Running/extending the evaluation harness |
