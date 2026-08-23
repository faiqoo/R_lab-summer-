"""Tests for the evaluation harness plumbing.

No real LLM/network calls: RAGComponents, HAWKIEvalModel construction, and
deepeval's evaluate() are all monkeypatched. Real metric-quality runs are a
manual `make eval` step against the live VM stack, not part of this fast
suite (LLM-judge scores are inherently non-deterministic).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("deepeval")

from archival_rag.config import Settings  # noqa: E402
from archival_rag.evaluation import dataset as dataset_mod  # noqa: E402
from archival_rag.evaluation import harness as harness_mod  # noqa: E402
from archival_rag.evaluation.models import EvalGolden  # noqa: E402
from archival_rag.models import RAGResult  # noqa: E402


def _settings(**overrides: Any) -> Settings:
    return Settings(
        hawki_token="test-token", _env_file=None, rag_eval_max_concurrent=1, **overrides
    )


class FakeHAWKIEvalModel:
    def __init__(self, settings: Settings) -> None:
        self._name = settings.rag_eval_model

    def get_model_name(self) -> str:
        return self._name


class FakeRAG:
    """Stands in for RAGComponents: returns canned results keyed by question."""

    def __init__(self, results: dict[str, RAGResult]) -> None:
        self._results = results
        self.closed = False
        self.calls: list[str] = []

    async def aquery(self, question: str, session_id: str) -> RAGResult:
        self.calls.append(question)
        return self._results[question]

    async def aclose(self) -> None:
        self.closed = True


def _answered(text: str = "chunk text") -> RAGResult:
    return RAGResult(
        answer="Die Antwort.",
        sources=[],
        trace={"declined": False, "retrieved_texts": [text]},
    )


def _declined() -> RAGResult:
    return RAGResult(answer="fallback", sources=[], trace={"declined": True})


async def test_expect_decline_goldens_skip_deepeval_and_update_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goldens = [
        EvalGolden(input="in-scope question", expect_decline=False),
        EvalGolden(input="out-of-scope question", expect_decline=True),
    ]
    fake_rag = FakeRAG(
        {
            "in-scope question": _answered(),
            "out-of-scope question": _declined(),
        }
    )
    monkeypatch.setattr(harness_mod, "RAGComponents", lambda settings: fake_rag)
    monkeypatch.setattr(harness_mod, "HAWKIEvalModel", FakeHAWKIEvalModel)

    called_with: dict[str, Any] = {}

    def fake_run_deepeval(**kwargs: Any) -> Any:
        called_with.update(kwargs)
        return type("R", (), {"test_results": []})()

    monkeypatch.setattr(harness_mod, "run_deepeval", fake_run_deepeval)

    report = await harness_mod.evaluate_rag(_settings(), goldens, metrics=[])

    assert fake_rag.closed is True
    assert report.guard_accuracy.true_negative == 1
    assert report.guard_accuracy.true_positive == 1
    assert report.guard_accuracy.false_negative == 0
    assert report.guard_accuracy.false_positive == 0
    # Only the in-scope, non-declined golden reaches deepeval.
    assert len(called_with["test_cases"]) == 1
    assert called_with["test_cases"][0].input == "in-scope question"
    assert called_with["test_cases"][0].retrieval_context == ["chunk text"]


async def test_guard_false_negative_and_false_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goldens = [
        EvalGolden(input="should decline but answered", expect_decline=True),
        EvalGolden(input="should answer but declined", expect_decline=False),
    ]
    fake_rag = FakeRAG(
        {
            "should decline but answered": _answered(),
            "should answer but declined": _declined(),
        }
    )
    monkeypatch.setattr(harness_mod, "RAGComponents", lambda settings: fake_rag)
    monkeypatch.setattr(harness_mod, "HAWKIEvalModel", FakeHAWKIEvalModel)
    monkeypatch.setattr(
        harness_mod,
        "run_deepeval",
        lambda **kw: type("R", (), {"test_results": []})(),
    )

    report = await harness_mod.evaluate_rag(_settings(), goldens, metrics=[])

    assert report.guard_accuracy.false_negative == 1
    assert report.guard_accuracy.false_positive == 1
    assert report.guard_accuracy.true_positive == 0
    assert report.guard_accuracy.true_negative == 0


async def test_limit_truncates_goldens(monkeypatch: pytest.MonkeyPatch) -> None:
    goldens = [EvalGolden(input=f"q{i}", expect_decline=True) for i in range(5)]
    fake_rag = FakeRAG({f"q{i}": _declined() for i in range(5)})
    monkeypatch.setattr(harness_mod, "RAGComponents", lambda settings: fake_rag)
    monkeypatch.setattr(harness_mod, "HAWKIEvalModel", FakeHAWKIEvalModel)

    report = await harness_mod.evaluate_rag(_settings(), goldens, metrics=[], limit=2)

    assert report.golden_count == 2
    assert len(fake_rag.calls) == 2


def _fake_metric_data(name: str, score: float, success: bool) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(name=name, score=score, success=success, reason="why")


async def test_checkpoint_written_during_run_and_cleared_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / ".checkpoint.json"
    goldens = [EvalGolden(input="q1"), EvalGolden(input="q2")]
    fake_rag = FakeRAG({"q1": _answered("c1"), "q2": _answered("c2")})
    monkeypatch.setattr(harness_mod, "RAGComponents", lambda settings: fake_rag)
    monkeypatch.setattr(harness_mod, "HAWKIEvalModel", FakeHAWKIEvalModel)

    def fake_run_deepeval(**kwargs: Any) -> Any:
        results = [
            type(
                "R",
                (),
                {
                    "success": True,
                    "metrics_data": [_fake_metric_data("Answer Relevancy", 0.9, True)],
                },
            )()
            for _ in kwargs["test_cases"]
        ]
        return type("Res", (), {"test_results": results})()

    monkeypatch.setattr(harness_mod, "run_deepeval", fake_run_deepeval)

    report = await harness_mod.evaluate_rag(
        _settings(), goldens, metrics=[], checkpoint_path=checkpoint_path
    )

    assert report.guard_accuracy.true_negative == 2
    assert len(report.golden_results) == 2
    # Cleared on full success — a later run shouldn't silently resume stale
    # cached results from a completed run.
    assert not checkpoint_path.exists()


async def test_resumes_from_checkpoint_skips_already_processed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / ".checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "generated": {
                    "q1": {
                        "answer": "cached answer",
                        "declined": False,
                        "retrieved_texts": ["c1"],
                    }
                },
                "scored": {
                    "q1": {
                        "success": True,
                        "scores": {"Answer Relevancy": 0.8},
                        "reasons": {},
                        "passes": {"Answer Relevancy": True},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    goldens = [EvalGolden(input="q1"), EvalGolden(input="q2")]
    # Only q2 is in the fake RAG's canned results — if q1 got re-queried
    # (i.e. resume didn't skip it), this raises KeyError and the test fails.
    fake_rag = FakeRAG({"q2": _answered("c2")})
    monkeypatch.setattr(harness_mod, "RAGComponents", lambda settings: fake_rag)
    monkeypatch.setattr(harness_mod, "HAWKIEvalModel", FakeHAWKIEvalModel)

    judged_inputs: list[str] = []

    def fake_run_deepeval(**kwargs: Any) -> Any:
        judged_inputs.extend(tc.input for tc in kwargs["test_cases"])
        results = [
            type(
                "R",
                (),
                {
                    "success": True,
                    "metrics_data": [_fake_metric_data("Answer Relevancy", 0.7, True)],
                },
            )()
            for _ in kwargs["test_cases"]
        ]
        return type("Res", (), {"test_results": results})()

    monkeypatch.setattr(harness_mod, "run_deepeval", fake_run_deepeval)

    report = await harness_mod.evaluate_rag(
        _settings(), goldens, metrics=[], checkpoint_path=checkpoint_path
    )

    assert fake_rag.calls == ["q2"]  # q1 skipped, only q2 actually queried
    assert judged_inputs == ["q2"]  # q1's cached score reused, not rejudged
    assert report.guard_accuracy.true_negative == 2
    assert len(report.golden_results) == 2
    scores_by_input = {
        r.input: r.scores["Answer Relevancy"] for r in report.golden_results
    }
    assert scores_by_input == {"q1": 0.8, "q2": 0.7}


def test_dataset_load_merges_anchor_and_synthesized(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.json"
    synth = tmp_path / "synth.json"
    missing_bibb = tmp_path / "no_bibb.json"
    anchor.write_text(
        json.dumps([{"input": "anchor q", "expected_output": "a"}]), encoding="utf-8"
    )
    synth.write_text(
        json.dumps([{"input": "synth q", "expected_output": "s"}]), encoding="utf-8"
    )

    goldens = dataset_mod.load_goldens(
        anchor_path=anchor, synthesized_path=synth, bibb_path=missing_bibb
    )

    assert [(g.input, g.provenance) for g in goldens] == [
        ("anchor q", "anchor"),
        ("synth q", "synthesized"),
    ]


def test_dataset_load_tolerates_missing_synthesized_file(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps([{"input": "anchor q"}]), encoding="utf-8")
    missing = tmp_path / "does_not_exist.json"
    missing_bibb = tmp_path / "no_bibb.json"

    goldens = dataset_mod.load_goldens(
        anchor_path=anchor, synthesized_path=missing, bibb_path=missing_bibb
    )

    assert len(goldens) == 1
    assert goldens[0].provenance == "anchor"


def test_dataset_load_merges_bibb(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.json"
    missing_synth = tmp_path / "no_synth.json"
    bibb = tmp_path / "bibb.json"
    anchor.write_text(json.dumps([{"input": "anchor q"}]), encoding="utf-8")
    bibb.write_text(
        json.dumps([{"input": "bibb q", "expected_output": "b"}]), encoding="utf-8"
    )

    goldens = dataset_mod.load_goldens(
        anchor_path=anchor, synthesized_path=missing_synth, bibb_path=bibb
    )

    assert [(g.input, g.provenance) for g in goldens] == [
        ("anchor q", "anchor"),
        ("bibb q", "bibb"),
    ]


def test_dataset_load_requires_anchor_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        dataset_mod.load_goldens(
            anchor_path=tmp_path / "missing.json",
            synthesized_path=tmp_path / "also_missing.json",
            bibb_path=tmp_path / "bibb_missing.json",
        )


def test_save_goldens_round_trips(tmp_path: Path) -> None:
    goldens = [
        EvalGolden(input="q1", expected_output="a1", context=["c1"]),
        EvalGolden(input="q2", expect_decline=True),
    ]
    out = tmp_path / "out.json"

    dataset_mod.save_goldens(goldens, out)
    loaded = dataset_mod._load_json_goldens(out, provenance="synthesized")

    assert [g.input for g in loaded] == ["q1", "q2"]
    assert loaded[1].expect_decline is True


def _report_dict(mean_score: float, accuracy: float | None) -> dict[str, Any]:
    return {
        "timestamp": "2026-01-01T00:00:00",
        "golden_count": 1,
        "judge_model": "test-model",
        "guard_accuracy": {
            "true_positive": 1,
            "false_negative": 0,
            "false_positive": 0,
            "true_negative": 0,
            "accuracy": accuracy,
            "precision": None,
            "recall": None,
        },
        "metric_summaries": [
            {
                "name": "Faithfulness",
                "mean_score": mean_score,
                "pass_rate": 1.0,
                "count": 1,
            }
        ],
        "golden_results": [],
    }


def test_compare_prints_deltas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_report_dict(0.70, 1.0)), encoding="utf-8")
    candidate.write_text(json.dumps(_report_dict(0.85, 0.5)), encoding="utf-8")

    harness_mod.compare(baseline, candidate)

    out = capsys.readouterr().out
    assert "Faithfulness" in out
    assert "0.7" in out and "0.85" in out
    assert "+0.15" in out or "0.15" in out
