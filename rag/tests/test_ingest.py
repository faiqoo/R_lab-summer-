"""Tests for year extraction precedence in ingestion."""

from __future__ import annotations

from archival_rag.ingestion.ingest import _extract_year


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
