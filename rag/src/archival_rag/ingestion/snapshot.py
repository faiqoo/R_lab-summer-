"""
Qdrant collection snapshot export / import.

Export (run once after ingestion, share the .snapshot file with colleagues):
    uv run python -m archival_rag.ingestion.snapshot export --out chunks.snapshot

Import (colleagues run this on their machine):
    uv run python -m archival_rag.ingestion.snapshot import --file chunks.snapshot
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx
from loguru import logger

from archival_rag.config import Settings
from archival_rag.logging_setup import setup_logging


def _base(settings: Settings) -> str:
    return f"http://{settings.qdrant_host}:{settings.qdrant_port}"


async def export(settings: Settings, out: Path) -> None:
    base = _base(settings)
    name = settings.qdrant_collection

    async with httpx.AsyncClient() as client:
        logger.info("Creating snapshot of '{}'...", name)
        r = await client.post(f"{base}/collections/{name}/snapshots", timeout=120)
        r.raise_for_status()
        snapshot_name: str = r.json()["result"]["name"]
        logger.info("Snapshot created: {}", snapshot_name)

        logger.info("Downloading to {} ...", out)
        async with client.stream(
            "GET", f"{base}/collections/{name}/snapshots/{snapshot_name}", timeout=300
        ) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with out.open("wb") as f:
                async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb_done = downloaded // 1_000_000
                        mb_total = total // 1_000_000
                        logger.debug("{} MB / {} MB ({:.0f}%)", mb_done, mb_total, pct)

    logger.info("Snapshot saved to: {}  ({} MB)", out, out.stat().st_size // 1_000_000)


async def import_(settings: Settings, file: Path) -> None:
    base = _base(settings)
    name = settings.qdrant_collection

    if not file.exists():
        raise SystemExit(f"File not found: {file}")

    logger.info(
        "Uploading {} ({} MB) to '{}'...", file, file.stat().st_size // 1_000_000, name
    )
    # Qdrant's upload endpoint only accepts multipart/form-data with the file in a
    # "snapshot" field; a raw application/octet-stream body is rejected with 415.
    async with httpx.AsyncClient() as client:
        with file.open("rb") as fh:
            r = await client.post(
                f"{base}/collections/{name}/snapshots/upload?priority=snapshot",
                files={"snapshot": (file.name, fh, "application/octet-stream")},
                timeout=600,
            )
        r.raise_for_status()
        logger.info("Collection '{}' restored.", name)
        info = (await client.get(f"{base}/collections/{name}")).json()["result"]
    logger.info("Points: {}  Status: {}", info["points_count"], info["status"])


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
    setup_logging(settings.log_level, service_name="snapshot")

    if args.cmd == "export":
        asyncio.run(export(settings, args.out))
    elif args.cmd == "import":
        asyncio.run(import_(settings, args.file))


if __name__ == "__main__":
    main()
