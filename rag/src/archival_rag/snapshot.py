"""
Qdrant collection snapshot export / import.

Export (run once after ingestion, share the .snapshot file with colleagues):
    uv run python -m archival_rag.snapshot export --out islandora_chunks.snapshot

Import (colleagues run this on their machine):
    uv run python -m archival_rag.snapshot import --file islandora_chunks.snapshot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from archival_rag.config import Settings


def _base(settings: Settings) -> str:
    return f"http://{settings.qdrant_host}:{settings.qdrant_port}"


def export(settings: Settings, out: Path) -> None:
    base = _base(settings)
    name = settings.qdrant_collection

    print(f"Creating snapshot of '{name}'...")
    r = httpx.post(f"{base}/collections/{name}/snapshots", timeout=120)
    r.raise_for_status()
    snapshot_name: str = r.json()["result"]["name"]
    print(f"  Snapshot created: {snapshot_name}")

    print(f"Downloading to {out} ...")
    with httpx.stream(
        "GET", f"{base}/collections/{name}/snapshots/{snapshot_name}", timeout=300
    ) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with out.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    mb_done = downloaded // 1_000_000
                    mb_total = total // 1_000_000
                    print(f"  {mb_done} MB / {mb_total} MB ({pct:.0f}%)", end="\r")
    print(f"\nDone. Snapshot saved to: {out}  ({out.stat().st_size // 1_000_000} MB)")


def import_(settings: Settings, file: Path) -> None:
    base = _base(settings)
    name = settings.qdrant_collection

    if not file.exists():
        raise SystemExit(f"File not found: {file}")

    print(f"Uploading {file} ({file.stat().st_size // 1_000_000} MB) to '{name}'...")
    with file.open("rb") as f:
        r = httpx.post(
            f"{base}/collections/{name}/snapshots/upload?priority=snapshot",
            content=f.read(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=600,
        )
    r.raise_for_status()
    print(f"Done. Collection '{name}' restored.")
    info = httpx.get(f"{base}/collections/{name}").json()["result"]
    print(f"  Points: {info['points_count']}  Status: {info['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qdrant snapshot export/import")
    sub = parser.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("export", help="Export collection to a .snapshot file")
    exp.add_argument(
        "--out",
        type=Path,
        default=Path("islandora_chunks.snapshot"),
        help="Output file path (default: islandora_chunks.snapshot)",
    )

    imp = sub.add_parser("import", help="Import a .snapshot file into Qdrant")
    imp.add_argument(
        "--file", type=Path, required=True, help="Path to the .snapshot file"
    )

    args = parser.parse_args()
    settings = Settings()

    if args.cmd == "export":
        export(settings, args.out)
    elif args.cmd == "import":
        import_(settings, args.file)


if __name__ == "__main__":
    main()
