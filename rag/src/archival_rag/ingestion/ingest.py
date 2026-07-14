"""
Ingestion pipeline: reads Ausbildungsordnung PDFs from a directory, chunks them
by section boundaries (text) and row boundaries (Anlage table), embeds each
chunk dense (bge-m3) + sparse (BM25), and upserts into Qdrant as named vectors
for hybrid search.

Usage:
    uv run python -m archival_rag.ingestion.ingest --pdf-dir /path/to/pdfs
    uv run python -m archival_rag.ingestion.ingest --pdf-dir /path/to/pdfs --recreate

Note: collections created before hybrid search (single unnamed vector) are
incompatible — re-ingest with --recreate (make ingest-recreate).
"""

from __future__ import annotations

import argparse
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # type: ignore[import-untyped]  # pymupdf has no stubs
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from archival_rag.config import Settings
from archival_rag.ingestion.extraction import (
    clean_page,
    detect_boilerplate,
    extract_page_text,
)
from archival_rag.retrieval.encoders import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    load_dense_encoder,
    load_sparse_encoder,
    to_sparse_vector,
)

# Stable namespace for deterministic chunk IDs (uuid5).
# Using uuid5(CHUNK_NS, "source::section::index") means re-running ingest
# without --recreate upserts (overwrites) instead of duplicating points.
_CHUNK_NS = uuid.UUID("b1a4f3c2-7d8e-4a0b-9c1d-2e5f6a7b8c9d")

# Matches section headers like "§ 1", "§1", "§ 12"
_SECTION_RE = re.compile(r"(?m)^\s*§\s*\d+")
# Matches the Anlage appendix header — handles "Anlage (zu § 5)" and "Anlage zu §" forms
_ANLAGE_RE = re.compile(r"Anlage\s*\(?\s*zu\s*§", re.IGNORECASE)
# BAAI/bge-m3 has max_seq_length=8192 tokens (~32k chars).
# 4000 chars (~1000 tokens) keeps chunks focused and well within the model limit.
_MAX_CHUNK_CHARS = 4000

# Ausbildungsrahmenplan item starts: "<Lfd. Nr.> <Capitalized word>" at line start
_TABLE_ITEM_RE = re.compile(r"\n(\d{1,3})\s+(?=[A-ZÄÖÜ])")
# The "(§ X Nr. Y)" citation that identifies a plan item across all document eras
_PARA_REF_RE = re.compile(r"\(§\s*(\d+)\s*Nr\.\s*(\d+[a-z]?)\)")

# Bundesanzeiger compilation detection: German "Ausbildungsprofil" page marker
_GERMAN_PROFILE_RE = re.compile(r"Ausbildungsprofil", re.IGNORECASE)
# Extracts the occupation name from "1 Berufsbezeichnung\n<Name>"
_BERUFSBEZEICHNUNG_RE = re.compile(r"1\s+Berufsbezeichnung\s*\n\s*([^\n\r]+)")


@dataclass
class Chunk:
    text: str
    chunk_type: str  # "text" | "table"
    section: str
    occupation: str
    year: int
    source: str


# ---------------------------------------------------------------------------
# Filename + text-based occupation/year extraction
# ---------------------------------------------------------------------------

# Ordered by era prevalence. Each pattern captures the occupation name in group 1.
_OCC_PATTERNS = [
    # DDR: "sozialistische Berufsbildung / Berufsausbildung\n<occupation>"
    re.compile(
        r"(?im)sozialistische\s+Berufs(?:bildung|ausbildung)\s*\n\s*([A-ZÄÖÜa-zäöüß][^\n]+)"
    ),
    # BRD: "Berufsausbildung zum X" (may have "/zur Y" after)
    re.compile(
        r"(?im)\bBerufsausbildung\s+zum[mr]?\s+([A-ZÄÖÜa-zäöüß][^\n/\(]+?)(?:/|\(|\n|$)"
    ),
    # Reich/Weimar all-caps one-liner: "BERUFSBILD DES/DER X■" or "BERUFSBILD DES/DER X"
    re.compile(r"(?m)^BERUFSBILD\s+(?:DES|DER)\s+([A-ZÄÖÜa-zäöüß][^\n■\(]+)"),
    # Mixed case one-liner: "Berufsbild des/der X"
    re.compile(
        r"(?im)\bBerufsbild\s+d(?:es|er)\s+([A-ZÄÖÜa-zäöüß][^\n\(■]+?)(?:\s*\(|$|\n)"
    ),
    # Multi-line DDR: "Berufsbild\n...\ndes\n...\n<occupation>" (up to 6 lines between)
    re.compile(
        r"(?im)\bBerufsbild\b(?:[^\n]*\n){1,6}d(?:es|er)\s+([A-ZÄÖÜa-zäöüß][^\n]+)"
    ),
    # Two-column scanned layout: "Berufsbild ...\n... des ...\n <Occupation> <junk>"
    # Stop capture at 3+ spaces (column separator) to avoid grabbing right-column noise.
    re.compile(
        r"(?im)\bBerufsbild\b(?:[^\n]*\n){1,4}[^\n]*\bdes\b[^\n]*\n\s*([A-ZÄÖÜa-zäöüß]\w+(?:\s+\w+){0,4})(?:\s{3,}|$|\n)"
    ),
    # "für den Ausbildungsberuf\n<occupation>"
    re.compile(r"(?im)für\s+den\s+Ausbildungsberuf\s*\n\s*([A-ZÄÖÜa-zäöüß][^\n]+)"),
    # "für den Lehrberuf\n<occupation>"
    re.compile(r"(?im)für\s+den\s+Lehrberuf\s*\n\s*([A-ZÄÖÜa-zäöüß][^\n]+)"),
    # Meister programme: "Meister für X"
    re.compile(r"(?im)\bMeister\s+für\s+([A-ZÄÖÜa-zäöüß][^\n]+)"),
]

# Prefer explicit date context ("vom 9. Juni 2005") over bare 4-digit years.
_YEAR_DATE_RE = re.compile(r"\bvom\s+\d+\.\s+\w+\s+(19\d{2}|20\d{2})\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _strip_genitive(name: str) -> str:
    """Strip German genitive -s/-es suffix: 'Drechslers' → 'Drechsler'."""
    # Only strip if the result is still a plausible word (≥ 4 chars)
    if name.endswith("ers") and len(name) > 5:
        return name[:-1]  # Asphaltierers → Asphaltierer
    if name.endswith("s") and not name.endswith("ss") and len(name) > 4:
        return name[:-1]  # Drechslers → Drechsler
    return name


def _parse_filename(path: Path) -> tuple[str, int]:
    """Return (occupation, year) from BRD filename like brd_bäcker_2004.pdf.

    Returns ('', 0) for berufearchiv_N.pdf files — caller extracts from text.
    """
    # macOS filenames are NFD (decomposed umlauts) — normalize so "für" is
    # one codepoint per letter and .title()/regexes behave.
    stem = unicodedata.normalize("NFC", path.stem)
    if not stem.startswith("brd_"):
        return "", 0
    year_match = re.search(r"_(\d{4})$", stem)
    year = int(year_match.group(1)) if year_match else 0
    occ = re.sub(r"^brd_", "", stem)
    occ = re.sub(r"_\d{4}$", "", occ)
    return occ.replace("_", " ").strip(), year


def _extract_from_text(text: str) -> tuple[str, int]:
    """Extract (occupation, year) from the first pages of a PDF."""
    occupation = ""
    for pattern in _OCC_PATTERNS:
        m = pattern.search(text)
        if m:
            raw = m.group(1).strip().rstrip("■.,;:")
            # Normalise all-caps to title case, then strip genitive suffix
            if raw.isupper():
                raw = raw.title()
            occupation = _strip_genitive(raw)
            break

    # Prefer an explicit date ("vom 9. Juni 2005") to avoid false positives
    # like Bundesanzeiger issue codes ("G 1990").
    m = _YEAR_DATE_RE.search(text) or _YEAR_RE.search(text)
    year = int(m.group(1)) if m else 0
    return occupation, year


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split prose at § boundaries. Returns [(label, content), ...]."""
    positions = [m.start() for m in _SECTION_RE.finditer(text)]
    if not positions:
        return [("text", text.strip())]

    chunks: list[tuple[str, str]] = []

    preamble = text[: positions[0]].strip()
    if preamble:
        chunks.append(("preamble", preamble))

    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        content = text[pos:end].strip()
        label_match = re.match(r"(§\s*\d+)", content)
        label = (
            label_match.group(1).replace(" ", " ")
            if label_match
            else f"section {i + 1}"
        )
        chunks.append((label, content))

    return chunks


def _split_by_pages(
    page_texts: list[str], occupation: str, year: int, source: str
) -> list[Chunk]:
    """Fallback: group pages into chunks that stay under _MAX_CHUNK_CHARS."""
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    group = 1

    for i, page_text in enumerate(page_texts):
        if current_len + len(page_text) > _MAX_CHUNK_CHARS and current:
            chunks.append(
                Chunk(
                    text="\n".join(current).strip(),
                    chunk_type="text",
                    section=f"pages {group}-{i}",
                    occupation=occupation,
                    year=year,
                    source=source,
                )
            )
            group = i + 1
            current = []
            current_len = 0
        current.append(page_text)
        current_len += len(page_text)

    if current:
        end = group + len(current) - 1
        label = f"page {group}" if group == end else f"pages {group}-{end}"
        chunks.append(
            Chunk(
                text="\n".join(current).strip(),
                chunk_type="text",
                section=label,
                occupation=occupation,
                year=year,
                source=source,
            )
        )

    return chunks


def _chunk_compilation(
    page_texts: list[str], profile_indices: list[int], base_year: int, source: str
) -> list[Chunk]:
    """Split a Bundesanzeiger multi-profile compilation into one chunk per occupation.

    Each German "Ausbildungsprofil" page is kept as a single chunk. The English and
    French translation pages that follow are dropped — bge-m3 handles cross-lingual
    retrieval natively, and including translations dilutes the embedding.
    """
    chunks: list[Chunk] = []
    for pos, page_idx in enumerate(profile_indices):
        page_text = page_texts[page_idx]
        m = _BERUFSBEZEICHNUNG_RE.search(page_text)
        if not m:
            continue
        # Strip trailing punctuation and soft-hyphen line-continuation artifacts (¬)
        occupation = m.group(1).strip().rstrip(".,;¬").rstrip()

        # German profile page only — English/French translations are redundant because
        # bge-m3 handles cross-lingual matching natively. Including them dilutes the
        # embedding and lowers retrieval scores for both German and English queries.
        group_text = page_texts[page_idx].strip()
        if not group_text:
            continue

        year = base_year
        if not year:
            ym = _YEAR_DATE_RE.search(page_text) or _YEAR_RE.search(page_text)
            year = int(ym.group(1)) if ym else 0

        chunks.append(
            Chunk(
                text=group_text[:_MAX_CHUNK_CHARS],
                chunk_type="text",
                section="Ausbildungsprofil",
                occupation=occupation,
                year=year,
                source=source,
            )
        )
    return chunks


def chunk_pdf(path: Path) -> list[Chunk]:
    """Parse a single PDF and return all chunks."""
    occupation, year = _parse_filename(path)
    doc = fitz.open(path)
    # sort=True fixes basic reading order without invoking Tesseract OCR.
    # The documents already have embedded text layers, so OCR is not needed.
    plain_texts = [page.get_text(sort=True) for page in doc]

    # berufearchiv_N.pdf filenames carry no metadata — extract from text.
    if not occupation:
        occupation, year = _extract_from_text("\n".join(plain_texts[:3]))

    # Bundesanzeiger compilations contain multiple occupations in one PDF.
    # Detect by finding 2+ German "Ausbildungsprofil" pages with Berufsbezeichnung.
    # Uses the raw text: their repeated per-profile structure ("1 Berufs-
    # bezeichnung" on every profile page) must not be boilerplate-stripped.
    if not occupation:
        profile_indices = [
            i
            for i, pt in enumerate(plain_texts)
            if _GERMAN_PROFILE_RE.search(pt) and _BERUFSBEZEICHNUNG_RE.search(pt)
        ]
        if len(profile_indices) >= 2:
            doc.close()
            return _chunk_compilation(plain_texts, profile_indices, year, path.name)

    # Find which page the Anlage starts on so we can label chunks correctly.
    anlage_start_page = None
    for i, pt in enumerate(plain_texts):
        if _ANLAGE_RE.search(pt):
            anlage_start_page = i
            break

    # Prose pages get column-aware extraction (older scans print §-sections in
    # two newspaper columns); Anlage pages print linearly and keep sort=True.
    page_texts = [
        (
            extract_page_text(doc[i])
            if anlage_start_page is None or i < anlage_start_page
            else plain_texts[i]
        )
        for i in range(len(plain_texts))
    ]
    doc.close()

    # Strip running headers/footers/page numbers detected across pages.
    boilerplate = detect_boilerplate(page_texts)
    page_texts = [clean_page(pt, boilerplate) for pt in page_texts]

    prose_pages = (
        page_texts[:anlage_start_page] if anlage_start_page is not None else page_texts
    )
    anlage_pages = (
        page_texts[anlage_start_page:] if anlage_start_page is not None else []
    )

    prose_text = "\n".join(prose_pages)
    chunks: list[Chunk] = []

    sections = _split_sections(prose_text)
    if any(len(content) > _MAX_CHUNK_CHARS for _, content in sections):
        # At least one section is too large: fall back to page-chunking the
        # whole prose exactly once. Calling _split_by_pages per oversized
        # section would duplicate the entire document.
        chunks.extend(_split_by_pages(prose_pages, occupation, year, path.name))
    else:
        for label, content in sections:
            if not content:
                continue
            chunks.append(
                Chunk(
                    text=content,
                    chunk_type="text",
                    section=label,
                    occupation=occupation,
                    year=year,
                    source=path.name,
                )
            )

    # Ausbildungsrahmenplan: one chunk per Lfd.-Nr. item, tagged with its
    # "(§ X Nr. Y)" citation — anchored on content, not table geometry.
    # Falls back to page-level chunks when no item pattern is found.
    anlage_text = "\n".join(anlage_pages)
    row_chunks = _chunk_anlage_rows(anlage_text, occupation, year, path.name)
    if row_chunks:
        chunks.extend(row_chunks)
    else:
        for i, page_text in enumerate(anlage_pages):
            if not page_text.strip():
                continue
            page_num = (anlage_start_page or 0) + i + 1
            label = f"Anlage page {page_num}"
            prefixed = (
                f"Document: Ausbildungsordnung {occupation.title()} {year}\n"
                f"Section: {label}\n\n{page_text}"
            )
            chunks.append(
                Chunk(
                    text=prefixed,
                    chunk_type="table",
                    section=label,
                    occupation=occupation,
                    year=year,
                    source=path.name,
                )
            )

    return chunks


def _chunk_anlage_rows(
    anlage_text: str, occupation: str, year: int, source: str
) -> list[Chunk]:
    """One chunk per Ausbildungsrahmenplan item ("Lfd. Nr.")."""
    text = "\n" + anlage_text
    matches = list(_TABLE_ITEM_RE.finditer(text))
    if not matches:
        return []

    chunks: list[Chunk] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.start() : end].strip()
        if not block:
            continue
        lfd_nr = m.group(1)

        para_match = _PARA_REF_RE.search(block)
        para_ref = (
            f"§ {para_match.group(1)} Nr. {para_match.group(2)}" if para_match else None
        )

        section = f"Anlage Nr. {lfd_nr}" + (f" ({para_ref})" if para_ref else "")
        header = (
            f"Document: Ausbildungsordnung {occupation.title()} {year}\n"
            f"Section: Ausbildungsrahmenplan Lfd. Nr. {lfd_nr}"
            + (f" ({para_ref})" if para_ref else "")
        )
        chunks.append(
            Chunk(
                text=f"{header}\n\n{block}"[:_MAX_CHUNK_CHARS],
                chunk_type="table",
                section=section,
                occupation=occupation,
                year=year,
                source=source,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------


def _ensure_collection(
    client: QdrantClient, name: str, dim: int, recreate: bool
) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if recreate and name in existing:
        client.delete_collection(name)
        existing.discard(name)
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=dim, distance=Distance.COSINE),
            },
            # BM25 term frequencies are computed client-side (fastembed);
            # Qdrant applies the IDF part server-side via the modifier.
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
            },
            # Higher m and ef_construct = better recall at query time as data grows.
            # m=32: more graph connections (default 16), better recall, ~2x memory.
            # ef_construct=200: more candidates during indexing (default 100).
            hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
        )


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------


_PROGRESS_EVERY = 500  # log a summary line every N PDFs


def embed_and_build_points(
    chunks: list[Chunk], model: Any, sparse_model: Any
) -> list[PointStruct]:
    """Embed chunks (dense + sparse) and build upsert-ready points.

    Deterministic uuid5 IDs: re-ingesting the same document overwrites its
    existing points instead of duplicating them.
    """
    texts = [c.text for c in chunks]
    vectors = model.encode(
        texts,
        show_progress_bar=False,
        batch_size=64,
    ).tolist()
    sparse_vectors = [to_sparse_vector(e) for e in sparse_model.passage_embed(texts)]

    return [
        PointStruct(
            id=str(
                uuid.uuid5(
                    _CHUNK_NS,
                    f"{chunk.source}::{chunk.section}::{idx}",
                )
            ),
            vector={
                DENSE_VECTOR_NAME: vec,
                SPARSE_VECTOR_NAME: sparse,
            },
            payload={
                "text": chunk.text,
                "chunk_type": chunk.chunk_type,
                "section": chunk.section,
                "occupation": chunk.occupation,
                "year": chunk.year,
                "source": chunk.source,
            },
        )
        for idx, (chunk, vec, sparse) in enumerate(zip(chunks, vectors, sparse_vectors))
    ]


def ingest_directory(pdf_dir: Path, settings: Settings, recreate: bool = False) -> None:
    model = load_dense_encoder(settings)
    sparse_model = load_sparse_encoder(settings)
    dim = model.get_embedding_dimension()
    if dim is None:
        raise SystemExit(
            f"Cannot determine embedding dimension for {settings.embedding_model}"
        )
    logger.info("Model ready: {}  dim={}", settings.embedding_model, dim)

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    _ensure_collection(client, settings.qdrant_collection, dim, recreate)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    total = len(pdfs)
    logger.info("Found {} PDFs in {}", total, pdf_dir)

    total_chunks = 0
    errors = 0
    no_occupation = 0
    no_year = 0

    for i, pdf_path in enumerate(pdfs, start=1):
        try:
            chunks = chunk_pdf(pdf_path)
            if not chunks:
                logger.debug("{}: skipped (no content)", pdf_path.name)
            else:
                if chunks[0].occupation == "":
                    no_occupation += 1
                if chunks[0].year == 0:
                    no_year += 1
                points = embed_and_build_points(chunks, model, sparse_model)
                client.upsert(collection_name=settings.qdrant_collection, points=points)
                total_chunks += len(chunks)
                logger.debug(
                    "{}: {} chunks  occ={}",
                    pdf_path.name,
                    len(chunks),
                    chunks[0].occupation or "?",
                )
        except Exception as exc:
            logger.error("{}: {}", pdf_path.name, exc)
            errors += 1

        if i % _PROGRESS_EVERY == 0 or i == total:
            logger.info(
                "Progress: {}/{} PDFs | {} chunks | {} errors",
                i,
                total,
                total_chunks,
                errors,
            )

    logger.info("Done. {} chunks indexed from {} PDFs.", total_chunks, total - errors)
    if errors:
        logger.warning("{} PDFs failed — check logs above.", errors)
    if no_occupation or no_year:
        logger.warning(
            "Data quality: {} PDFs with no occupation extracted, {} with no year.",
            no_occupation,
            no_year,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Ausbildungsordnung PDFs into Qdrant"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting",
    )
    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.pdf_dir}")

    settings = Settings()
    from archival_rag.logging_setup import setup_logging

    setup_logging(settings.log_level)
    ingest_directory(args.pdf_dir, settings, recreate=args.recreate)


if __name__ == "__main__":
    main()
