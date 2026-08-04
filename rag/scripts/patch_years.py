"""One-off: patch the stale `year` payload field on already-ingested Qdrant
points using the corrected Erlassdatum > Inkrafttreten > Erscheinungsjahr
precedence (`archival_rag.ingestion.ingest._extract_year`), without
re-embedding.

Chunking, section labels, and occupation extraction are unchanged — only
year extraction improved — so re-running `chunk_pdf()` on the same PDFs
yields the exact same deterministic point IDs already in Qdrant. This lets
us recompute just `year` per chunk and patch it in place via `set_payload`,
skipping the GPU/embedding pass entirely.

Run on the GPU VM, where the source PDFs live (~archivdaten) and Qdrant is
local:
    uv run --env-file .env python scripts/patch_years.py --pdf-dir ~/archivdaten
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path
from typing import Any, cast

from loguru import logger
from qdrant_client import AsyncQdrantClient

from archival_rag.config import Settings
from archival_rag.ingestion.ingest import _CHUNK_NS, chunk_pdf
from archival_rag.logging_setup import setup_logging

_SCROLL_PAGE_SIZE = 1000
_PATCH_BATCH = 500


async def _fetch_existing_years(
    client: AsyncQdrantClient, collection: str
) -> dict[str, int]:
    existing: dict[str, int] = {}
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=["year"],
            with_vectors=False,
        )
        for p in points:
            existing[str(p.id)] = (p.payload or {}).get("year", 0)
        if offset is None or not points:
            break
    return existing


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.pdf_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.pdf_dir}")

    settings = Settings()
    setup_logging(settings.log_level)
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    logger.info("Fetching existing years from {}...", settings.qdrant_collection)
    existing = await _fetch_existing_years(client, settings.qdrant_collection)
    logger.info("Existing points: {}", len(existing))

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    logger.info("Found {} PDFs in {}", len(pdfs), args.pdf_dir)

    # Group point IDs by their NEW year so each distinct year is one
    # set_payload call instead of one per chunk (~66k points but typically
    # only a couple hundred distinct years across the whole corpus).
    by_new_year: dict[int, list[str]] = {}
    total_chunks = 0
    changed = 0
    missing = 0
    errors = 0

    for i, pdf_path in enumerate(pdfs, start=1):
        try:
            chunks = chunk_pdf(pdf_path)
        except Exception as exc:
            logger.error("{}: {}", pdf_path.name, exc)
            errors += 1
            continue
        for idx, chunk in enumerate(chunks):
            point_id = str(
                uuid.uuid5(_CHUNK_NS, f"{chunk.source}::{chunk.section}::{idx}")
            )
            total_chunks += 1
            if point_id not in existing:
                missing += 1
                continue
            if existing[point_id] != chunk.year:
                changed += 1
                by_new_year.setdefault(chunk.year, []).append(point_id)
        if i % 500 == 0 or i == len(pdfs):
            logger.info(
                "Parsed {}/{} PDFs | {} chunks | {} changed so far",
                i,
                len(pdfs),
                total_chunks,
                changed,
            )

    logger.info(
        "Patching {} changed points across {} distinct year values...",
        changed,
        len(by_new_year),
    )
    for year, point_ids in by_new_year.items():
        for i in range(0, len(point_ids), _PATCH_BATCH):
            batch = point_ids[i : i + _PATCH_BATCH]
            await client.set_payload(
                collection_name=settings.qdrant_collection,
                payload={"year": year},
                # list[str] is a structural mismatch under mypy only because
                # list is invariant — set_payload accepts str IDs at runtime.
                points=cast(Any, batch),
            )

    logger.info(
        "Done. {} chunks parsed from {} PDFs, {} years patched, "
        "{} chunks had no matching point (corpus drift?), {} PDF errors.",
        total_chunks,
        len(pdfs) - errors,
        changed,
        missing,
        errors,
    )
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
