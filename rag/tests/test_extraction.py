"""Tests for page extraction/cleaning and Anlage row chunking."""

from __future__ import annotations

import fitz  # type: ignore[import-untyped]

from archival_rag.ingestion.extraction import (
    clean_page,
    detect_boilerplate,
    extract_page_text,
)
from archival_rag.ingestion.ingest import _chunk_anlage_rows


def test_detect_boilerplate_flags_repeating_lines() -> None:
    pages = [
        "Bundesgesetzblatt Jahrgang 1974\n§ 1 Inhalt eins\nSeite 1",
        "Bundesgesetzblatt Jahrgang 1974\n§ 2 Inhalt zwei\nSeite 2",
        "Bundesgesetzblatt Jahrgang 1974\n§ 3 Inhalt drei\nSeite 3",
    ]
    boilerplate = detect_boilerplate(pages)

    assert "Bundesgesetzblatt Jahrgang 1974" in boilerplate
    assert "§ 1 Inhalt eins" not in boilerplate


def test_clean_page_strips_noise() -> None:
    boilerplate = {"Bundesgesetzblatt Jahrgang 1974"}
    text = (
        "Bundesgesetzblatt Jahrgang 1974\n"
        "§ 1 Ausbildungsdauer\n"
        "1 2 3 4\n"
        "42\n"
        "Die Ausbildung dauert drei Jahre."
    )

    cleaned = clean_page(text, boilerplate)

    assert cleaned == "§ 1 Ausbildungsdauer\nDie Ausbildung dauert drei Jahre."


def test_extract_page_text_reconstructs_two_columns() -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Two-column layout with realistic geometry: the left column reaches into
    # the middle band (like real scans), so the gutter is detectable.
    for row, y in enumerate((100, 120, 140), start=1):
        page.insert_text((50, y), f"l{row}a")
        page.insert_text((150, y), f"l{row}b")
        page.insert_text((240, y), f"l{row}c")
        page.insert_text((330, y), f"r{row}a")
        page.insert_text((380, y), f"r{row}b")

    text = extract_page_text(page)
    doc.close()

    words = text.split()
    # Full left column must come before the right column.
    assert words.index("l3c") < words.index("r1a")


def test_anlage_rows_split_on_item_numbers() -> None:
    anlage = (
        "Abschnitt I Berufliche Grundbildung\n"
        "1 Berufsbildung (§ 4 Nr. 1)\n"
        "a) Bedeutung des Ausbildungsvertrages erklären\n"
        "2 Aufbau des Ausbildungsbetriebes (§ 4 Nr. 2)\n"
        "b) Grundfunktionen erläutern\n"
    )

    chunks = _chunk_anlage_rows(anlage, "bäcker", 1982, "brd_baecker_1982.pdf")

    assert len(chunks) == 2
    assert chunks[0].section == "Anlage Nr. 1 (§ 4 Nr. 1)"
    assert chunks[1].section == "Anlage Nr. 2 (§ 4 Nr. 2)"
    assert "Lfd. Nr. 1 (§ 4 Nr. 1)" in chunks[0].text
    assert "Ausbildungsvertrages" in chunks[0].text
    assert chunks[0].chunk_type == "table"
    assert chunks[0].occupation == "bäcker"


def test_anlage_rows_empty_when_no_items() -> None:
    assert _chunk_anlage_rows("Nur Fließtext ohne Nummern.", "x", 1990, "x.pdf") == []


def test_oversized_single_page_is_split() -> None:
    from archival_rag.ingestion.ingest import _MAX_CHUNK_CHARS, _split_by_pages

    huge_page = "\n".join(f"Zeile {i} mit etwas Inhalt." for i in range(600))
    assert len(huge_page) > _MAX_CHUNK_CHARS

    chunks = _split_by_pages([huge_page], "bäcker", 1980, "x.pdf")

    assert len(chunks) > 1
    assert all(len(c.text) <= _MAX_CHUNK_CHARS + 100 for c in chunks)
    assert chunks[0].section.startswith("page 1 part")


def test_parse_filename_normalizes_nfd_umlauts() -> None:
    import unicodedata
    from pathlib import Path

    from archival_rag.ingestion.ingest import _parse_filename

    nfd = unicodedata.normalize("NFD", "brd_bäcker_2004.pdf")
    occupation, year = _parse_filename(Path(nfd))
    assert occupation == "bäcker"
    assert year == 2004


def test_extract_from_text_finds_occupation_and_dated_year() -> None:
    from archival_rag.ingestion.ingest import _extract_from_text

    text = (
        "Verordnung über die Berufsausbildung zum Drechsler\n"
        "Vom 9. Juni 2005\n(BGBl. I S. 1581)\nG 1990 Ausgabe\n"
    )
    occupation, year = _extract_from_text(text)
    assert occupation == "Drechsler"
    # "vom 9. Juni 2005" must win over the "G 1990" issue-code false positive.
    assert year == 2005
