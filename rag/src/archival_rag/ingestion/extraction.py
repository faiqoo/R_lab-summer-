"""Page-text extraction and cleaning for the ingestion pipeline.

Column-aware extraction and automatic boilerplate removal adapted from
Tahira Ghaffar's chunking pipeline (pdf_to_chunks_v3.py, merged in
dd90e45), ported from pdfplumber word geometry to pymupdf so the
pipeline keeps a single PDF dependency.

Older Ausbildungsordnung scans print the prose §-sections in two
newspaper-style columns; plain extraction interleaves the columns line
by line and produces garbled chunks. ``extract_page_text`` detects the
column gutter from word geometry and reconstructs proper reading order
(left column top-to-bottom, then right column). Lines spanning the full
page width — titles, headings — stay in place.

``detect_boilerplate`` flags any line that repeats near-identically on
3+ pages (running headers, footers, repeated table column headers) so
``clean_page`` can strip them without per-document hardcoding.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, TypedDict


class _Word(TypedDict):
    x0: float
    x1: float
    top: float
    text: str


def _group_lines(words: list[_Word]) -> list[list[_Word]]:
    lines: list[list[_Word]] = []
    current: list[_Word] = []
    current_top: float | None = None
    for w in words:
        if current_top is None or abs(w["top"] - current_top) <= 2.5:
            current.append(w)
            if current_top is None:
                current_top = w["top"]
        else:
            lines.append(current)
            current, current_top = [w], w["top"]
    if current:
        lines.append(current)
    return lines


def _find_gutter(words: list[_Word], page_width: float) -> float:
    """Locate the column gutter: the widest x-gap in the middle band."""
    xs = sorted(w["x0"] for w in words)
    lo, hi = page_width * 0.35, page_width * 0.65
    best_gap, gutter = 0.0, page_width / 2
    for a, b in zip(xs, xs[1:]):
        if lo <= a <= hi and (b - a) > best_gap:
            best_gap, gutter = b - a, (a + b) / 2
    return gutter


def extract_page_text(page: Any) -> str:
    """Extract a page in reading order, reconstructing two-column layouts."""
    raw = page.get_text("words")
    if not raw:
        return str(page.get_text(sort=True))
    words: list[_Word] = sorted(
        ({"x0": w[0], "top": w[1], "x1": w[2], "text": w[4]} for w in raw),
        key=lambda w: (round(w["top"], 1), w["x0"]),
    )
    page_width = float(page.rect.width)
    gutter = _find_gutter(words, page_width)

    output: list[str] = []
    left_buffer: list[str] = []
    right_buffer: list[str] = []

    def flush() -> None:
        output.extend(left_buffer)
        output.extend(right_buffer)
        left_buffer.clear()
        right_buffer.clear()

    for line in _group_lines(words):
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
                # Full-width line (title/heading): emit buffered columns first.
                flush()
                output.append(text)
            elif x0 < gutter:
                left_buffer.append(text)
            else:
                right_buffer.append(text)

    flush()
    return "\n".join(output)


def detect_boilerplate(page_texts: list[str], min_repeats: int = 3) -> set[str]:
    """Lines repeating on ``min_repeats``+ pages are headers/footers."""
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
