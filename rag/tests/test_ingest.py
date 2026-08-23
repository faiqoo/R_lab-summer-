"""Tests for year extraction precedence and plain-text ingestion."""

from __future__ import annotations

from pathlib import Path

from archival_rag.ingestion.ingest import _extract_year, chunk_document


def test_erlassdatum_wins_over_inkrafttreten_and_bare_year() -> None:
    text = (
        "Verordnung vom 9. Juni 2005\n"
        "Diese Verordnung tritt am 1. August 2006 in Kraft.\n"
        "Stand: 1990"
    )
    assert _extract_year(text) == 2005


def test_inkrafttreten_wins_over_bare_year_when_no_erlassdatum() -> None:
    text = "Diese Verordnung tritt am 1. August 2006 in Kraft.\nStand: 1990"
    assert _extract_year(text) == 2006


def test_inkrafttreten_getreten_phrasing() -> None:
    text = "Die Regelung ist in Kraft getreten am 1. Januar 1990."
    assert _extract_year(text) == 1990


def test_falls_back_to_bare_year_when_no_dated_phrase() -> None:
    text = "Bundesanzeiger Jahrgang 1990, Nr. 42\nStand: 1990"
    assert _extract_year(text) == 1990


def test_returns_zero_when_no_year_present() -> None:
    assert _extract_year("Kein Datum hier.") == 0


def test_chunk_document_reads_txt(tmp_path: Path) -> None:
    """RS.IG0020: .txt files chunk the same way PDFs do."""
    path = tmp_path / "brd_bäcker_2004.txt"
    path.write_text(
        "§ 1 Ausbildungsberuf\nDer Beruf wird anerkannt.\n"
        "§ 2 Ausbildungsdauer\nDie Ausbildung dauert 3 Jahre.",
        encoding="utf-8",
    )

    chunks = chunk_document(path)

    # Section labels use a non-breaking space (_split_sections), same as PDFs.
    assert len(chunks) == 2
    assert [c.section for c in chunks] == ["§\xa01", "§\xa02"]
    assert chunks[0].occupation == "bäcker"
    assert chunks[0].year == 2004
    assert chunks[0].source == "brd_bäcker_2004.txt"


def test_chunk_document_txt_splits_on_form_feed(tmp_path: Path) -> None:
    """Form feeds are the plain-text page-break convention — honor them so
    per-page logic (boilerplate detection) sees real pages."""
    path = tmp_path / "brd_maurer_1985.txt"
    path.write_text("§ 1 Erste Seite\nInhalt eins.\f§ 2 Zweite Seite\nInhalt zwei.")

    chunks = chunk_document(path)

    assert [c.section for c in chunks] == ["§\xa01", "§\xa02"]
