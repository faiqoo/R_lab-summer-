.PHONY: install test lint format check ingest ingest-recreate export import-snapshot embed-service tunnel api flow-deploy flow-test

PDF_DIR ?= ../archivdaten
SNAPSHOT_FILE ?= islandora_chunks.snapshot

# conda's site.py skips .pth files starting with _ (including the editable install
# _editable_impl_archival_rag.pth written by uv pip install -e .).
# PYTHONPATH=src is the portable fix that works regardless of venv/conda setup.
export PYTHONPATH := $(CURDIR)/src

install:
	uv sync --dev && uv pip install -e . && uv run pre-commit install

test:
	uv run pytest -v

lint:
	uv run mypy src/ && uv run ruff check src/ tests/ && uv run black --check src/ tests/

format:
	uv run black src/ tests/ && uv run ruff check --fix src/ tests/

check: lint test

# --- Data ---

ingest:
	uv run --env-file .env python -m archival_rag.ingestion.ingest --pdf-dir $(PDF_DIR)

ingest-recreate:
	uv run --env-file .env python -m archival_rag.ingestion.ingest --pdf-dir $(PDF_DIR) --recreate

export:
	uv run --env-file .env python -m archival_rag.ingestion.snapshot export --out $(SNAPSHOT_FILE)

import-snapshot:
	uv run --env-file .env python -m archival_rag.ingestion.snapshot import --file $(SNAPSHOT_FILE)

# --- Services ---

# RAG service (ragsvc) — the API exposed to the Drupal chatbot (/ingest, /health, chat soon).
api:
	uv run --env-file .env uvicorn archival_rag.api:app --host 0.0.0.0 --port 8000

# Embedding service (loads the model locally — run on the GPU VM or for offline dev).
# On the VM this runs under systemd (embedsvc.service); this target is for manual runs.
embed-service:
	uv run --env-file .env uvicorn archival_rag.embedding_service:app --host 0.0.0.0 --port 8080

# SSH tunnel to the GPU VM services (needs uni VPN + rag/.env.vm).
# VM Qdrant -> localhost:16333, embedding service -> localhost:8080.
tunnel:
	@. ./.env.vm && exec ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
		-L 16333:localhost:6333 -L 8080:localhost:8080 $$GPU_VM_USER@$$GPU_VM_HOST

# --- Node-RED indexing flow (ActiveMQ -> ragsvc /ingest) ---

# Deploy/refresh the flow into the running Node-RED container (port 1881).
flow-deploy:
	@. ./.env.vm && RAGSVC_KEY=$$EMBEDDING_API_KEY uv run python scripts/deploy_nodered_flow.py

# Fire a synthetic Islandora event through ActiveMQ to test the whole path.
# Needs: activemq+nodered containers up, ragsvc running (make api), and a
# file server for the test PDF (see isle-dc/build/nodered-data/README.md).
flow-test:
	docker cp scripts/test_event.json isle-dc-activemq-1:/tmp/event.json
	docker exec isle-dc-activemq-1 /opt/activemq/bin/activemq producer \
		--destination queue://islandora-connector-rag --messageCount 1 \
		--persistent true --payloadUrl file:///tmp/event.json
