# Running the Full Stack

Two separate places things run, connected by the uni VPN tunnel:

```text
Your laptop / isle-dc                      GPU VM (141.26.157.121)
┌─────────────────────────┐                ┌──────────────────────────┐
│ Drupal (isle-dc compose) │                │ qdrant.service            │
│ ActiveMQ (isle-dc compose)│               │   (vector DB)             │
│ Node-RED (isle-dc compose)│──── VPN ─────▶│ embedsvc.service          │
│                          │   tunnel/      │   (bge-m3 on GPU)         │
│ your dev tools (chat,    │   or direct    │ ragsvc.service            │
│  search_repl, eval)      │───once IT ────▶│   (podman container —     │
└─────────────────────────┘   opens 8000)   │    the one Drupal calls)  │
                                            └──────────────────────────┘
```

## GPU VM side: one command starts everything

The three VM services now have real dependency ordering (`ragsvc.service`
`Wants=` + `After=` the other two, and waits for both to actually respond
before starting — not just "process launched"). So:

```bash
ssh lab25@<GPU_VM_HOST>
systemctl --user start ragsvc     # pulls up qdrant + embedsvc first, then itself
systemctl --user status qdrant embedsvc ragsvc
curl localhost:8000/health
```

They're also all `WantedBy=default.target` with lingering enabled, so a full
VM reboot brings all three back up in the correct order automatically —
nothing to do by hand after a restart.

Unit files are version-controlled at `rag/deploy/systemd/*.service` (source
of truth) — if the VM is ever rebuilt, copy them to
`~/.config/systemd/user/` there and `systemctl --user daemon-reload`.

### Restarting one service without the others

```bash
systemctl --user restart ragsvc      # code change to the API — safe, doesn't touch qdrant/embedsvc
systemctl --user restart embedsvc    # model/embedding change — do NOT run a bulk ingest at the same time (10GB GPU, see gpu-vm-services.md)
systemctl --user restart qdrant      # rare — only if the DB itself is misbehaving
```

## Your laptop side: the tunnel

Everything on your laptop (dev chat testing, `search_repl.py`, the eval
harness, `make ingest`) reaches the VM through one SSH tunnel:

```bash
cd rag && make tunnel   # keep running in its own terminal
```

Maps `localhost:16333` → VM Qdrant, `localhost:8080` → VM embedding service,
and port 8000 → VM `ragsvc`. The local isle-dc Drupal calls
`host.docker.internal:8000`, so this tunnel is all the chatbot needs. Port 8000
is bound on all interfaces because on Linux/WSL the Drupal container reaches
the host through the Docker bridge, not loopback. Stop `make api` before
starting the tunnel, it uses the same port.
See [gpu-vm-services.md](gpu-vm-services.md) for the full `.env` setup.

## isle-dc side (Drupal + ActiveMQ + Node-RED)

This is a **separate** docker-compose stack, unrelated to the VM's podman
setup — see the root `isle-dc/` docs for bringing it up
(`docker compose up -d activemq nodered`, etc.). The Node-RED ingest flow
(`isle-dc/build/nodered-data/rag-ingest-flow.json`) is what actually calls
`ragsvc`'s `/ingest` — deploy/test it per
[the flow's own README](../../isle-dc/build/nodered-data/README.md)
(`make flow-deploy`, `make flow-test`).

## ⚠️ The one real gap: Drupal reaching ragsvc for real

Node-RED (running locally in isle-dc) and any real Drupal instance both need
to reach `ragsvc` at `http://<GPU_VM_HOST>:8000` over the network. Today
that only works through an SSH tunnel (dev-only) — **the VM firewall allows
SSH inbound only**, so a real Drupal deployment cannot reach it until port
8000 is opened for Drupal's origin. That's a university IT request, not
something fixable from this account. See
[deploying-ragsvc.md](deploying-ragsvc.md) for the specifics — confirm where
Drupal will actually run and get that firewall change requested before
treating this as production-ready end to end.

## Quick "is everything up" check

```bash
# From your laptop, tunnel running:
curl localhost:16333            && echo "Qdrant OK"
curl localhost:8080/health      && echo "Embedding service OK"
curl localhost:8000/health      && echo "ragsvc OK"
```
