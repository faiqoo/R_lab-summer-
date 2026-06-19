# Contributing

## Branch Structure

```
main
└── feature/rag          ← integration branch — ALL PRs target this
    ├── your-feature-1
    ├── your-feature-2
    └── ...
```

**Never open a PR directly to `main`.** All work goes through `feature/rag` first. `main` only receives a merge from `feature/rag` once the integration is stable and reviewed.

## Opening a PR

1. Branch off `feature/rag`:
   ```bash
   git checkout feature/rag
   git pull
   git checkout -b your-name/short-description
   ```

2. Do your work, commit, push.

3. Open a PR with **base: `feature/rag`** (not `main`).

## Teams

| Team | Scope |
|------|-------|
| RAG / Python | `rag/` — LangChain wrapper, embedding service, RAG orchestrator |
| Drupal | Chatbot UI module inside the Drupal container |

Both teams PR into `feature/rag`. The Drupal team and RAG team work independently — coordinate on the HTTP interface between the Chatbot UI and the RAG Chatbot Service.

## RAG Project Setup

See [`rag/README.md`](rag/README.md) for Python environment setup (`uv`, pre-commit hooks, running tests).

## Code Quality

Pre-commit hooks are enforced on every commit in `rag/`:

```bash
cd rag
make install   # first time only
make check     # mypy + ruff + black + pytest
```

All checks must pass before requesting review.
