# Deploying ragsvc (the API the Drupal team calls)

`ragsvc` is the one HTTP surface Drupal's chatbot module talks to:
`/chat`, `/chat/stream`, `/ingest`, `/ingest/async`, `/status/{id}`,
`/health`. Drupal never talks to Qdrant or the embedding service directly —
`ragsvc` is the only door.

Runs as a container on the GPU VM, alongside the existing systemd-managed
Qdrant (`qdrant.service`) and embedding service (`embedsvc.service`), reached
over the host network so it can hit them at `localhost:6333`/`localhost:8080`
exactly like it would running bare-metal.

## Why podman, not docker

The VM has no Docker (confirmed 2026-07-28) and `lab25` has no sudo to
install it. **Rootless podman is already installed and working** — the
standard fix for exactly this situation. There's also no
docker-compose/podman-compose provider on the VM, so deployment there is a
plain `podman build` + `podman run`, wrapped in a systemd `--user` unit —
the same pattern already used for Qdrant and the embedding service, not a
compose file. (`docker-compose.yml` in this repo is for local/dev
convenience only — see the comment at its top.)

## One-time setup on the VM

```bash
# From your laptop: sync the current source (as always)
rsync -a --exclude .venv --exclude "__pycache__" --exclude ".env*" \
  --exclude "*.snapshot" --exclude "eval_runs" \
  rag/ lab25@<GPU_VM_HOST>:rag/

# On the VM: build the image (torch download makes this slow — several minutes)
ssh lab25@<GPU_VM_HOST>
cd rag && podman build -t archival-rag-ragsvc .
```

`rag/.env` on the VM needs (separate from your laptop's tunnel-mode `.env` —
on the VM itself, Qdrant/embedding-service are local, not tunneled):

```bash
HAWKI_TOKEN=<the real token>       # chat endpoints return 503 without this
QDRANT_HOST=localhost
QDRANT_PORT=6333                  # NOT 16333 — that remap is only for the laptop tunnel
EMBEDDING_API_URL=http://localhost:8080
EMBEDDING_API_KEY=<from .env.vm>
```

Systemd unit (`~/.config/systemd/user/ragsvc.service` on the VM):

```ini
[Unit]
Description=archival-rag ragsvc (RAG API for Drupal)
After=network.target

[Service]
WorkingDirectory=%h/rag
ExecStartPre=-/usr/bin/podman rm -f ragsvc
ExecStart=/usr/bin/podman run --rm --name ragsvc --network host --env-file %h/rag/.env archival-rag-ragsvc
ExecStop=/usr/bin/podman stop -t 10 ragsvc
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now ragsvc
curl localhost:8000/health
```

## Redeploying after a code change

```bash
rsync -a ... rag/ lab25@<GPU_VM_HOST>:rag/
ssh lab25@<GPU_VM_HOST> 'cd rag && podman build -t archival-rag-ragsvc . && systemctl --user restart ragsvc'
```

## ⚠️ Network reachability — needs a decision before Drupal can actually call this

The VM's firewall allows **SSH only** (confirmed earlier this session — no
sudo to change it). That means:

- **From your laptop, over the VPN tunnel**: works today, same pattern as
  Qdrant/embedding-service (`ssh -L 8000:localhost:8000 ...`). Fine for dev.
- **From wherever Drupal actually runs, over a normal network call**: will
  **not** work until port 8000 is opened on the VM's firewall for Drupal's
  origin, which needs university IT (a firewall change, not something this
  account can do). Verify where the Drupal instance the team will use is
  actually hosted, then request that opening — don't assume this works
  end-to-end until that's confirmed and tested from Drupal's actual network.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `curl localhost:8000/health` hangs on the VM itself | Container not running: `systemctl --user status ragsvc`, check `journalctl --user -u ragsvc` |
| `503` from `/chat` | `HAWKI_TOKEN` missing from the VM's `rag/.env` |
| Container can't reach Qdrant/embedding-service | Confirm `--network host` is set and `QDRANT_PORT=6333` (not 16333) in the VM's `.env` |
| Reachable via tunnel but not directly | Expected — see the firewall note above; needs a university IT firewall change |
