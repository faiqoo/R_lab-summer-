# Islandora RAG Chatbot — Summer Lab 2026

Research lab project (Summer 2026, Uni Koblenz, Institute for Software Technology) extending the **Berufearchiv** Islandora digital archive with an LLM-powered chatbot that can answer questions over digitised archival documents.

## What We're Building

A Retrieval-Augmented Generation (RAG) chatbot layered on top of an existing Islandora 2.x repository. Users ask natural-language questions through a Drupal-embedded chat interface; the system retrieves relevant document chunks from a vector database and feeds them to an LLM to generate a grounded answer.

## Architecture

```
Researcher
    │
    ▼
Drupal (Chatbot UI)
    │ HTTP
    ▼
RAG Chatbot Service  ──► Embedding Service ──► Qdrant (vector search)
    │
    ▼
HAWKI LLM API (qwen3-30b-a3b-instruct-2507)
```

**Indexing path (event-driven):**
Upload → Drupal → ActiveMQ → Node-RED → Embedding Service → Qdrant

**Query path (synchronous):**
User → Chatbot UI → RAG Service → embed query → Qdrant similarity search → LLM → answer

Full C4 model: [`docs/workspace.dsl`](docs/workspace.dsl) (open with [Structurizr](https://structurizr.com/dsl) or Structurizr Lite).

## Repository Layout

```
docs/           C4 architecture model (Structurizr DSL)
isle-dc/        Islandora Docker Compose stack (existing system)
rag/            Python RAG project (new — this is where we build)
scripts/        Utility scripts
```

## Key Technology Choices

| Component | Technology | Reason |
|-----------|-----------|--------|
| LLM | HAWKI API (`chat.edu-ki-rlp.de`) | University-provided LLM gateway, no infra cost |
| Default model | `qwen3-30b-a3b-instruct-2507` | Best instruction-following on HAWKI |
| Vector DB | Qdrant | Built-in BM25 hybrid search, container-friendly, dashboard UI |
| Embedding | sentence-transformers | Lightweight, runs locally |
| RAG orchestration | LangChain + FastAPI | Industry standard, flexible |
| Python tooling | uv + pyproject.toml | Fast, reproducible installs |

## Getting Started

### Existing Islandora stack

```bash
cd isle-dc
docker compose up -d
```

### RAG project

```bash
cd rag
cp .env.example .env       # fill in HAWKI_TOKEN
make install               # uv sync + pre-commit hooks
make test                  # run the test suite
```

See [`rag/README.md`](rag/README.md) for full usage docs and examples.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branching model and PR workflow.
All PRs target `feature/rag` — not `main`.

## Team

- RAG / Python team — `rag/` directory
- Drupal team — Chatbot UI module
