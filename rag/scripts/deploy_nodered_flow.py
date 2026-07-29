"""Deploy the RAG ingest flow into a running Node-RED via its admin API.

Replaces any previously deployed version of the flow (matched by tab id),
optionally injecting the ragsvc API key from the environment so the repo
copy of the flow never contains secrets.

Usage:
    RAGSVC_KEY=<key> python scripts/deploy_nodered_flow.py [--url http://localhost:1881]
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

FLOW_FILE = (
    Path(__file__).resolve().parents[2]
    / "isle-dc"
    / "build"
    / "nodered-data"
    / "rag-ingest-flow.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:1881")
    args = parser.parse_args()

    flow = json.loads(FLOW_FILE.read_text())
    key = os.environ.get("RAGSVC_KEY", "")
    if key:
        for node in flow:
            if node.get("type") == "function":
                node["func"] = node["func"].replace(
                    "env.get('RAGSVC_KEY') || ''",
                    f"env.get('RAGSVC_KEY') || '{key}'",
                )

    flow_ids = {n["id"] for n in flow}
    tab_ids = {n["id"] for n in flow if n.get("type") == "tab"}

    with urllib.request.urlopen(f"{args.url}/flows", timeout=10) as resp:
        existing = json.load(resp)

    # Drop any previous version of this flow (tab members + same-id nodes).
    kept = [
        n for n in existing if n["id"] not in flow_ids and n.get("z") not in tab_ids
    ]
    merged = kept + flow

    body = json.dumps(merged).encode()
    req = urllib.request.Request(
        f"{args.url}/flows",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Node-RED-Deployment-Type": "full",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(f"deployed rag-ingest-flow to {args.url} (HTTP {resp.status})")


if __name__ == "__main__":
    main()
