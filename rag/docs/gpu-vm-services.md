# Using the GPU VM Services (Embedding Service + Qdrant)

The university GPU VM (A100 slice, 10GB VRAM) hosts the two shared services of
the RAG system. You develop on your laptop; the heavy lifting happens there:

| Service | On the VM | Via tunnel on your laptop |
| --- | --- | --- |
| Embedding service (bge-m3, GPU) | `localhost:8080` | `http://localhost:8080` |
| Qdrant (65k+ chunk collection) | `localhost:6333` | `http://localhost:16333` |

Both run as systemd services and survive reboots — you never need to start
them, only connect to them. No model weights are ever loaded on your laptop.

## Prerequisites (once)

1. **Uni VPN access** working (the VM is unreachable without it).
2. **Credentials**: ask Ashhad for the `rag/.env.vm` file (VM host/user/password
   + `EMBEDDING_API_KEY`). It is gitignored — never commit it.
3. **SSH key** (recommended, so you don't type the password constantly):

   ```bash
   ssh-copy-id lab25@<GPU_VM_HOST>   # host/user/password from .env.vm
   ```

4. Project set up as usual: `cd rag && make install`.

## Every work session

```bash
# 1. Connect the uni VPN.

# 2. Open the tunnel (keep this terminal running):
cd rag
make tunnel

# 3. Verify (in a second terminal):
curl http://localhost:8080/health
#   -> {"model":"BAAI/bge-m3","dim":1024}
curl http://localhost:16333
#   -> {"title":"qdrant - vector search engine", ...}
```

Your `rag/.env` needs these three lines (values for the key from `.env.vm`):

```bash
QDRANT_PORT=16333
EMBEDDING_API_URL=http://localhost:8080
EMBEDDING_API_KEY=<ask Ashhad>
```

That's it. Any code using `load_dense_encoder(settings)` or `HybridSearcher`
now embeds on the VM's GPU and searches the shared collection automatically —
no code changes.

Port 16333 (not 6333) because your local Docker Qdrant may occupy 6333; the
tunnel maps the VM's Qdrant next to it.

## Trying it directly

```bash
# Embed a text (X-API-Key from .env.vm):
curl -X POST http://localhost:8080/embed \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <key>' \
  -d '{"inputs": ["Wie lange dauert die Ausbildung zum Bäcker?"]}'

# Browse the collection in the Qdrant dashboard:
open http://localhost:16333/dashboard
```

Interactive API docs (Swagger): <http://localhost:8080/docs>

## Working offline (train, no VPN)

Remove/comment the three `.env` lines above → the code falls back to a local
model and your local Qdrant on 6333. Import the collection snapshot once
(`islandora_chunks.snapshot`, ask Ashhad) with:

```bash
make import-snapshot SNAPSHOT_FILE=islandora_chunks.snapshot
```

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `curl` hangs / connection refused | VPN not connected, or tunnel not running (`make tunnel`) |
| `401 invalid or missing API key` | `EMBEDDING_API_KEY` missing/wrong in `.env` |
| Tunnel exits immediately | Another process on local port 8080/16333 — free it first |
| `/health` down but tunnel fine | Service crashed on the VM: `ssh lab25@<host> 'systemctl --user restart embedsvc'` (Qdrant: `... restart qdrant`) |
| Slow first request after restart | Model loading (~30 s) — retry |

## Rules

- **Never run a bulk ingest against the VM while `embedsvc` is running** — they
  can't share the 10GB GPU. Coordinate with Ashhad; the procedure is: stop
  `embedsvc`, run the ingest on the VM, start `embedsvc`.
- The collection `islandora_chunks` is shared team state — don't delete or
  recreate it without telling everyone.
- Don't commit `.env`, `.env.vm`, or any snapshot files (all gitignored).
