"""Evaluation harness: run the golden dataset through the real RAG chain,
score it with deepeval RAG metrics (judged by HAWKI), and report results.

Usage (module: archival_rag.evaluation.harness):
    uv run --group eval --env-file .env python -m archival_rag.evaluation.harness \\
        run [--limit N] | synthesize | compare <baseline.json> <candidate.json>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.evaluate import evaluate as run_deepeval
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from loguru import logger

from archival_rag.chain import RAGComponents
from archival_rag.config import Settings
from archival_rag.evaluation.dataset import load_goldens
from archival_rag.evaluation.judge import HAWKIEvalModel
from archival_rag.evaluation.models import (
    EvalGolden,
    EvalReport,
    GoldenResult,
    GuardAccuracy,
    MetricSummary,
)
from archival_rag.logging_setup import setup_logging

_REPO_ROOT = Path(__file__).resolve().parents[3]


def default_metrics(judge: HAWKIEvalModel) -> list[BaseMetric]:
    return [
        AnswerRelevancyMetric(model=judge, include_reason=True),
        FaithfulnessMetric(model=judge, include_reason=True),
        ContextualPrecisionMetric(model=judge, include_reason=True),
        ContextualRecallMetric(model=judge, include_reason=True),
        ContextualRelevancyMetric(model=judge, include_reason=True),
    ]


async def evaluate_rag(
    settings: Settings,
    goldens: list[EvalGolden],
    metrics: list[BaseMetric] | None = None,
    limit: int | None = None,
) -> EvalReport:
    """Run every golden through RAGComponents.aquery(), score with deepeval.

    Deliberately calls aquery() for EVERY golden (including expect_decline
    ones) to observe the guard's real behavior, then splits: goldens the
    guard declined (or that were expected to decline) go to GuardAccuracy
    only — scoring generation quality against a refusal, or against chunks
    the guard itself judged too weak, is meaningless. Only true negatives
    (expected an answer, guard agreed) go to the deepeval metric suite.
    """
    if limit is not None:
        goldens = goldens[:limit]

    judge = HAWKIEvalModel(settings)
    metrics = metrics if metrics is not None else default_metrics(judge)
    rag = RAGComponents(settings)

    guard = GuardAccuracy()
    test_cases: list[LLMTestCase] = []
    case_meta: list[EvalGolden] = []

    # Bounded-concurrency generation: a plain sequential loop is fine at
    # tens of goldens but becomes the bottleneck at hundreds (each aquery()
    # is a full embed + hybrid-search + HAWKI-generation round trip). Reuses
    # rag_eval_max_concurrent so one setting governs total HAWKI load across
    # both this phase and deepeval's own metric-judging concurrency below.
    semaphore = asyncio.Semaphore(settings.rag_eval_max_concurrent)

    async def _run_one(golden: EvalGolden) -> tuple[EvalGolden, Any]:
        async with semaphore:
            session_id = f"eval-{uuid.uuid4().hex[:8]}"
            result = await rag.aquery(golden.input, session_id=session_id)
            return golden, result

    try:
        pairs = await asyncio.gather(*(_run_one(g) for g in goldens))
    finally:
        await rag.aclose()

    for golden, result in pairs:
        declined = bool(result.trace.get("declined", False))

        if golden.expect_decline:
            if declined:
                guard.true_positive += 1
            else:
                guard.false_negative += 1
            continue
        if declined:
            guard.false_positive += 1
            continue
        guard.true_negative += 1

        retrieval_context = cast("list[str]", result.trace.get("retrieved_texts", []))
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=result.answer,
                # LLMTestCase's retrieval_context is list[str |
                # RetrievedContextData] | None; list invariance makes our
                # plain list[str] a structural mismatch under mypy even
                # though it's exactly what deepeval expects at runtime.
                retrieval_context=cast(Any, retrieval_context),
                expected_output=golden.expected_output,
            )
        )
        case_meta.append(golden)

    golden_results: list[GoldenResult] = []
    metric_totals: dict[str, list[float]] = {}
    metric_passes: dict[str, list[bool]] = {}

    if test_cases:
        eval_result = run_deepeval(
            test_cases=test_cases,
            metrics=metrics,
            async_config=AsyncConfig(max_concurrent=settings.rag_eval_max_concurrent),
            display_config=DisplayConfig(
                print_results=False, show_indicator=False, inspect_after_run=False
            ),
            error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
        )
        for golden, test_result in zip(case_meta, eval_result.test_results):
            scores: dict[str, float] = {}
            reasons: dict[str, str] = {}
            for md in test_result.metrics_data or []:
                if md.score is not None:
                    scores[md.name] = md.score
                    metric_totals.setdefault(md.name, []).append(md.score)
                    metric_passes.setdefault(md.name, []).append(md.success)
                if md.reason:
                    reasons[md.name] = md.reason
            golden_results.append(
                GoldenResult(
                    input=golden.input,
                    provenance=golden.provenance,
                    success=test_result.success,
                    scores=scores,
                    reasons=reasons,
                )
            )

    summaries = [
        MetricSummary(
            name=name,
            mean_score=sum(scores) / len(scores),
            pass_rate=sum(metric_passes[name]) / len(metric_passes[name]),
            count=len(scores),
        )
        for name, scores in metric_totals.items()
    ]

    return EvalReport(
        timestamp=datetime.now(UTC).isoformat(),
        golden_count=len(goldens),
        judge_model=judge.get_model_name(),
        guard_accuracy=guard,
        metric_summaries=summaries,
        golden_results=golden_results,
    )


def _report_to_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "timestamp": report.timestamp,
        "golden_count": report.golden_count,
        "judge_model": report.judge_model,
        "guard_accuracy": asdict(report.guard_accuracy)
        | {
            "accuracy": report.guard_accuracy.accuracy,
            "precision": report.guard_accuracy.precision,
            "recall": report.guard_accuracy.recall,
        },
        "metric_summaries": [asdict(m) for m in report.metric_summaries],
        "golden_results": [asdict(g) for g in report.golden_results],
    }


def save_report(report: EvalReport, settings: Settings) -> Path:
    out_dir = _REPO_ROOT / settings.rag_eval_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = report.timestamp.replace(":", "-")
    path = out_dir / f"{ts}.json"
    path.write_text(
        json.dumps(_report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def print_report(report: EvalReport) -> None:
    print(
        f"\n=== Eval report ({report.golden_count} goldens, "
        f"judge={report.judge_model}) ==="
    )
    g = report.guard_accuracy
    print(
        f"Guard accuracy: {g.accuracy} (tp={g.true_positive} fn={g.false_negative} "
        f"fp={g.false_positive} tn={g.true_negative})"
    )
    for anchor_only in (True, False):
        provenance = "anchor" if anchor_only else "synthesized"
        rows = report.summary_by(provenance)
        if not rows:
            continue
        print(f"\n--- {provenance} ({len(rows)} goldens) ---")
        for r in rows:
            print(f"  [{'OK' if r.success else 'FAIL'}] {r.input[:70]}")
            for name, score in r.scores.items():
                print(f"      {name}: {score:.2f}")
    print("\n--- Metric summary (all goldens) ---")
    for m in report.metric_summaries:
        print(
            f"  {m.name}: mean={m.mean_score:.3f} "
            f"pass_rate={m.pass_rate:.2f} (n={m.count})"
        )


def compare(baseline_path: Path, candidate_path: Path) -> None:
    """Human-eyeball diff between two report JSON files — not a CI gate.

    LLM-judge scores are non-deterministic; this highlights deltas for
    manual review, it does not assert pass/fail.
    """
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    print(f"Comparing {baseline_path.name} -> {candidate_path.name}")

    base_guard = baseline["guard_accuracy"]
    cand_guard = candidate["guard_accuracy"]
    print("\nGuard accuracy:")
    for key in ("accuracy", "precision", "recall"):
        b, c = base_guard.get(key), cand_guard.get(key)
        delta = "" if b is None or c is None else f" ({c - b:+.3f})"
        print(f"  {key}: {b} -> {c}{delta}")

    base_metrics = {m["name"]: m["mean_score"] for m in baseline["metric_summaries"]}
    cand_metrics = {m["name"]: m["mean_score"] for m in candidate["metric_summaries"]}
    print("\nPer-metric mean score:")
    for name in sorted(set(base_metrics) | set(cand_metrics)):
        b, c = base_metrics.get(name), cand_metrics.get(name)
        delta = "" if b is None or c is None else f" ({c - b:+.3f})"
        print(f"  {name}: {b} -> {c}{delta}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation harness (deepeval)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run the golden dataset through the RAG chain")
    run.add_argument("--limit", type=int, default=None, help="Only use first N goldens")

    synth = sub.add_parser("synthesize", help="Regenerate goldens_synthesized.json")
    synth.add_argument(
        "--n-groups",
        type=int,
        default=200,
        help="Distinct source documents to sample from (the real breadth lever)",
    )
    synth.add_argument(
        "--chunks-per-group", type=int, default=4, help="Chunks per document context"
    )
    synth.add_argument(
        "--max-goldens-per-context",
        type=int,
        default=3,
        help="Questions generated per document",
    )

    cmp_parser = sub.add_parser("compare", help="Diff two report JSON files")
    cmp_parser.add_argument("baseline", type=Path)
    cmp_parser.add_argument("candidate", type=Path)

    args = parser.parse_args()
    settings = Settings()
    setup_logging(settings.log_level)

    if args.cmd == "run":
        goldens = load_goldens()
        report = asyncio.run(evaluate_rag(settings, goldens, limit=args.limit))
        print_report(report)
        path = save_report(report, settings)
        logger.info("Report saved: {}", path)
    elif args.cmd == "synthesize":
        from archival_rag.evaluation.dataset import synthesize
        from archival_rag.retrieval.search import HybridSearcher

        async def _run() -> None:
            searcher = HybridSearcher(settings)
            try:
                await synthesize(
                    settings,
                    searcher,
                    n_groups=args.n_groups,
                    chunks_per_group=args.chunks_per_group,
                    max_goldens_per_context=args.max_goldens_per_context,
                )
            finally:
                await searcher.aclose()

        asyncio.run(_run())
    elif args.cmd == "compare":
        compare(args.baseline, args.candidate)


if __name__ == "__main__":
    main()
