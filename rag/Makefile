.PHONY: install install-eval test lint format check ingest ingest-recreate export import-snapshot embed-service tunnel api flow-deploy flow-test eval eval-dataset eval-compare docker-build docker-run vm-stack-status vm-deploy-ragsvc

PDF_DIR ?= ../archivdaten
SNAPSHOT_FILE ?= islandora_chunks.snapshot

# conda's site.py skips .pth files starting with _ (including the editable install
# _editable_impl_archival_rag.pth written by uv pip install -e .).
# PYTHONPATH=src is the portable fix that works regardless of venv/conda setup.
export PYTHONPATH := $(CURDIR)/src

install:
	uv sync --dev && uv pip install -e . && uv run pre-commit install

# Evaluation tooling (deepeval) is heavy (pandas/rich/tabulate) and eval-only —
# kept out of the default install. See docs/evaluation.md.
install-eval:
	uv sync --dev --group eval && uv pip install -e .

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

# RAG service (ragsvc) — the API exposed to the Drupal chatbot (/chat, /ingest, /health).
api:
	uv run --env-file .env uvicorn archival_rag.api:app --host 0.0.0.0 --port 8000

# Container build/run for local testing. VM deployment uses podman directly
# (no compose provider there) — see docs/deploying-ragsvc.md.
docker-build:
	docker build -t archival-rag-ragsvc .

docker-run:
	docker compose up --build

# --- GPU VM stack (Qdrant + embedsvc + ragsvc) — see docs/running-the-full-stack.md ---

# Status of all three VM services + a live health check, over the tunnel/SSH.
vm-stack-status:
	@. ./.env.vm && ssh $$GPU_VM_USER@$$GPU_VM_HOST 'systemctl --user status qdrant embedsvc ragsvc --no-pager -l | grep -E "●|Active:"; echo; curl -s localhost:8000/health && echo'

# Sync source + rebuild + restart ragsvc on the VM (code changes only —
# qdrant/embedsvc are untouched). Needs uni VPN.
vm-deploy-ragsvc:
	@. ./.env.vm && rsync -a --exclude .venv --exclude "__pycache__" --exclude ".env*" \
		--exclude "*.snapshot" --exclude eval_runs . $$GPU_VM_USER@$$GPU_VM_HOST:rag/
	@. ./.env.vm && ssh $$GPU_VM_USER@$$GPU_VM_HOST 'cd rag && podman build -t archival-rag-ragsvc . && systemctl --user restart ragsvc'

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

# --- Evaluation (deepeval; make install-eval first — see docs/evaluation.md) ---

# Run the golden dataset through the real RAG chain, score with deepeval RAG
# metrics judged by HAWKI, write a timestamped report to eval_runs/.
# Cheap smoke check: make eval ARGS="--limit 5"
eval:
	uv run --group eval --env-file .env python -m archival_rag.evaluation.harness run $(ARGS)

# Regenerate goldens_synthesized.json from the live Qdrant collection.
# Review the output before committing — synthesized expected_output is
# LLM-generated and can be wrong.
eval-dataset:
	uv run --group eval --env-file .env python -m archival_rag.evaluation.harness synthesize

# Human-eyeball diff between two eval_runs/*.json reports — not a CI gate.
# Usage: make eval-compare BASELINE=eval_runs/a.json CANDIDATE=eval_runs/b.json
eval-compare:
	uv run --group eval --env-file .env python -m archival_rag.evaluation.harness compare $(BASELINE) $(CANDIDATE)
