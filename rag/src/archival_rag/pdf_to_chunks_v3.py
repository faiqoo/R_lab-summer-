"""
PDF -> chunks. Plain-language explanation of how this works:

WHY THIS VERSION IS DIFFERENT
------------------------------
Earlier versions tried to detect the TABLE'S COLUMNS (which cell holds
the item number, which holds the description, etc). That breaks the
moment a PDF's table is laid out differently -- which turned out to be
true even within your own corpus (a 1974 doc and a 2002 doc had
genuinely different column arrangements).

This version does NOT look at table columns at all. Instead it uses
something that stays constant across your whole corpus regardless of
layout: every training-content item is written as

    <item number>  <category name>  ...  (§ <para> Nr. <n>)  ...content...

and the next item starts the same way. So instead of parsing table
geometry, we just read the page as plain text (in natural reading
order) and cut a new chunk every time we see "<number> <Capitalized
word>" at the start of a line. This is far more robust to layout
differences because it's anchored to the DOCUMENT'S CONTENT, not its
visual formatting.

STEPS
-----
1. Read every page as plain text (pdfplumber's extract_text).
2. Auto-detect and remove boilerplate: any line that repeats near-
   identically across 3+ pages (running headers, footers, repeated
   table column headers, "1 2 3 4" column-index rows). This is
   detected automatically -- nothing hardcoded per document.
3. Decide, page by page, whether it belongs to the prose section or
   the appendix/table section (looks for the 'Anlage' / '(zu §'
   heading that marks the start of the appendix across the corpus).
4. Prose: split into chunks at "§ N" section headings.
5. Table: split into chunks every time a new "<number> <Word>" item
   start is found. Each chunk also gets tagged with whichever
   "(§ X Nr. Y)" citation appears inside it, since that citation is
   the one thing guaranteed to identify the item consistently across
   every era of this document format.

Install once:  pip install pdfplumber
Run:           python pdf_to_chunks_v3.py your_file.pdf
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from collections import Counter
from typing import Any

import pdfplumber

Chunk = dict[str, Any]

PARA_REF = re.compile(r"\(§\s*(\d+)\s*Nr\.\s*(\d+[a-z]?)\)")
TABLE_ITEM_START = re.compile(r"\n(\d{1,3})\s+(?=[A-ZÄÖÜ])")
PROSE_SECTION_START = re.compile(r"\n§\s*(\d+[a-z]?)\s*\n")


# --------------------------------------------------------------
# 1) Read PDF, page by page
#
# Some pages (usually the prose §-sections) are printed in two
# newspaper-style columns. Reading these with plain extract_text()
# scrambles left/right column text together on the same line.
# extract_columns_aware() detects the column gutter and reconstructs
# proper reading order: full left column top-to-bottom, then full
# right column top-to-bottom. Lines that genuinely span the whole
# page width (titles, headings) are kept intact.
# --------------------------------------------------------------


def extract_columns_aware(
    page: Any, gutter_frac: tuple[float, float] = (0.35, 0.65)
) -> str:
    words = page.extract_words()
    if not words:
        return str(page.extract_text() or "")

    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines: list[list[dict[str, Any]]] = []
    current_line: list[dict[str, Any]] = []
    current_top: float | None = None
    for w in words:
        if current_top is None or abs(w["top"] - current_top) <= 2.5:
            current_line.append(w)
            current_top = w["top"] if current_top is None else current_top
        else:
            lines.append(current_line)
            current_line, current_top = [w], w["top"]
    if current_line:
        lines.append(current_line)

    xs = sorted(w["x0"] for w in words)
    lo, hi = page.width * gutter_frac[0], page.width * gutter_frac[1]
    best_gap, gutter = 0.0, page.width / 2
    for a, b in zip(xs, xs[1:]):
        if lo <= a <= hi and (b - a) > best_gap:
            best_gap = b - a
            gutter = (a + b) / 2

    output: list[str] = []
    left_buffer: list[str] = []
    right_buffer: list[str] = []

    def flush() -> None:
        output.extend(left_buffer)
        output.extend(right_buffer)
        left_buffer.clear()
        right_buffer.clear()

    for line in lines:
        line_sorted = sorted(line, key=lambda w: w["x0"])
        split_idx: int | None = None
        for i in range(len(line_sorted) - 1):
            gap = line_sorted[i + 1]["x0"] - line_sorted[i]["x1"]
            mid = (line_sorted[i + 1]["x0"] + line_sorted[i]["x1"]) / 2
            if gap > 15 and (gutter - 40) <= mid <= (gutter + 40):
                split_idx = i + 1
                break

        if split_idx is not None:
            left_buffer.append(" ".join(w["text"] for w in line_sorted[:split_idx]))
            right_buffer.append(" ".join(w["text"] for w in line_sorted[split_idx:]))
        else:
            text = " ".join(w["text"] for w in line_sorted)
            x0, x1 = line_sorted[0]["x0"], line_sorted[-1]["x1"]
            if x0 < gutter - 20 and x1 > gutter + 20:
                flush()
                output.append(text)
            elif x0 < gutter:
                left_buffer.append(text)
            else:
                right_buffer.append(text)

    flush()
    return "\n".join(output)


def page_is_table_page(plain_text: str) -> bool:
    """A page belongs to the appendix/table once we've seen the
    'Anlage' heading followed shortly by '(zu §' -- a stable marker
    for this section across the corpus."""
    return bool(re.search(r"Anlage\s*\n\(zu\s*§", plain_text))


def read_pages(pdf_path: str) -> tuple[list[str], list[bool]]:
    """
    For each page, decide once whether it's a TABLE page (appendix) or
    a PROSE page, then extract with whichever method suits that page:
      - table pages: plain extract_text() (verified correct reading
        order against real corpus data -- these tables print in a
        naturally linear order even though visually columnar)
      - prose pages: column-aware reconstruction (fixes two-column
        newspaper-style layout scrambling)
    Returns one combined, per-page text list plus a matching list of
    booleans marking which pages are table pages.
    """
    with pdfplumber.open(pdf_path) as pdf:
        plain_pages = [(p.extract_text() or "") for p in pdf.pages]

        seen_table_heading = False
        page_texts: list[str] = []
        is_table: list[bool] = []
        for page, plain in zip(pdf.pages, plain_pages):
            if not seen_table_heading and page_is_table_page(plain):
                seen_table_heading = True
            if seen_table_heading:
                page_texts.append(plain)
            else:
                page_texts.append(extract_columns_aware(page))
            is_table.append(seen_table_heading)

    return page_texts, is_table


# --------------------------------------------------------------
# 2) Auto-detect boilerplate lines (headers/footers/repeated table
#    headers) -- no hardcoded patterns, works across document eras
# --------------------------------------------------------------


def detect_boilerplate(page_texts: list[str], min_repeats: int = 3) -> set[str]:
    counts: Counter[str] = Counter()
    for pt in page_texts:
        seen: set[str] = set()
        for line in pt.split("\n"):
            line = line.strip()
            if line and line not in seen:
                counts[line] += 1
                seen.add(line)
    return {line for line, c in counts.items() if c >= min_repeats}


def clean_page(text: str, boilerplate: set[str]) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s or s in boilerplate:
            continue
        if re.fullmatch(r"(\d\s*){2,}", s):  # "1 2 3 4" column-index rows
            continue
        if re.fullmatch(r"\d{1,4}", s):  # bare page numbers
            continue
        out.append(s)
    return "\n".join(out)


# --------------------------------------------------------------
# 4) Prose chunking by "§ N" heading
# --------------------------------------------------------------


def chunk_prose(text: str, regulation_name: str, document_date: str) -> list[Chunk]:
    text = "\n" + text
    marks = list(PROSE_SECTION_START.finditer(text))
    if not marks:
        if text.strip():
            return [
                _chunk(
                    text.strip(),
                    "text_section",
                    {
                        "section_number": None,
                        "regulation_name": regulation_name,
                        "document_date": document_date,
                    },
                )
            ]
        return []

    out: list[Chunk] = []
    for i, m in enumerate(marks):
        start = m.start()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        piece = text[start:end].strip()
        if piece:
            out.append(
                _chunk(
                    piece,
                    "text_section",
                    {
                        "section_number": f"§ {m.group(1)}",
                        "regulation_name": regulation_name,
                        "document_date": document_date,
                    },
                )
            )
    return out


# --------------------------------------------------------------
# 5) Table chunking by content-anchored item boundaries
# --------------------------------------------------------------


def chunk_table(text: str, regulation_name: str, document_date: str) -> list[Chunk]:
    matches = list(TABLE_ITEM_START.finditer(text))
    if not matches:
        return []

    out: list[Chunk] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        lfd_nr = m.group(1)

        para_match = PARA_REF.search(block)
        para_ref = (
            f"§ {para_match.group(1)} Nr. {para_match.group(2)}" if para_match else None
        )

        header = (
            f"[{regulation_name}, {document_date}] "
            f"Ausbildungsrahmenplan, Lfd. Nr. {lfd_nr}"
        )
        if para_ref:
            header += f" ({para_ref})"

        full_text = f"{header}:\n{block}"

        out.append(
            _chunk(
                full_text,
                "table_row",
                {
                    "lfd_nr": lfd_nr,
                    "paragraph_ref": para_ref,
                    "regulation_name": regulation_name,
                    "document_date": document_date,
                },
            )
        )
    return out


def _chunk(text: str, chunk_type: str, metadata: dict[str, Any]) -> Chunk:
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        "chunk_type": chunk_type,
        "metadata": metadata,
    }


# --------------------------------------------------------------
# MAIN -- the only function you call
# --------------------------------------------------------------


def chunk_pdf(
    pdf_path: str, regulation_name: str = "", document_date: str = ""
) -> list[Chunk]:
    page_texts, is_table = read_pages(pdf_path)
    boilerplate = detect_boilerplate(page_texts)
    cleaned_pages = [clean_page(pt, boilerplate) for pt in page_texts]

    prose_text = "\n".join(pt for pt, t in zip(cleaned_pages, is_table) if not t)
    table_text = "\n".join(pt for pt, t in zip(cleaned_pages, is_table) if t)

    prose_chunks = chunk_prose(prose_text, regulation_name, document_date)
    table_chunks = chunk_table(table_text, regulation_name, document_date)

    return prose_chunks + table_chunks


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "example.pdf"
    reg_name = sys.argv[2] if len(sys.argv) > 2 else ""
    doc_date = sys.argv[3] if len(sys.argv) > 3 else ""

    chunks = chunk_pdf(path, regulation_name=reg_name, document_date=doc_date)

    print(f"Generated {len(chunks)} chunks\n")
    for c in chunks:
        print(f"--- {c['chunk_type']} | {c['metadata']} ---")
        print(c["text"][:200], "\n")

    with open("chunks_output.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("Saved all chunks -> chunks_output.json")
