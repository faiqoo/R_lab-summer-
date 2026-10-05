"""Evaluation harness: run the golden dataset through the real RAG chain,
score it with deepeval RAG metrics (judged by HAWKI), and report results.

Usage (module: scripts.evaluation.harness):
    uv run --group eval --env-file .env python -m scripts.evaluation.harness \\
        run [--limit N] | synthesize | compare <baseline.json> <candidate.json>
"""

from __future__ import annotations

import argparse
import asyncio
import functools
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
from archival_rag.logging_setup import setup_logging
from scripts.evaluation.dataset import load_goldens
from scripts.evaluation.judge import HAWKIEvalModel
from scripts.evaluation.models import (
    EvalGolden,
    EvalReport,
    GoldenResult,
    GuardAccuracy,
    MetricSummary,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKPOINT_PATH = _REPO_ROOT / "eval_runs" / ".checkpoint.json"
# Judging is batched (a) so a crash mid-run only loses one batch's worth of
# work instead of everything since the last checkpoint had no granularity
# at all, and (b) to give visible progress during deepeval's otherwise
# silent judging phase (its own indicator is disabled — see below).
_JUDGE_BATCH_SIZE = 50
# How often the generation phase logs/checkpoints — generation is fast
# enough that per-golden checkpointing would be excessive I/O.
_GENERATE_CHECKPOINT_EVERY = 50


def default_metrics(judge: HAWKIEvalModel) -> list[BaseMetric]:
    return [
        AnswerRelevancyMetric(model=judge, include_reason=True),
        FaithfulnessMetric(model=judge, include_reason=True),
        ContextualPrecisionMetric(model=judge, include_reason=True),
        ContextualRecallMetric(model=judge, include_reason=True),
        ContextualRelevancyMetric(model=judge, include_reason=True),
    ]


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {"generated": {}, "scored": {}}
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("generated", {})
        data.setdefault("scored", {})
        return cast("dict[str, dict[str, Any]]", data)
    except Exception:
        logger.warning("Checkpoint at {} unreadable — starting fresh", path)
        return {"generated": {}, "scored": {}}


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    # Write-then-rename so a crash mid-write can't corrupt the checkpoint
    # a subsequent run would otherwise try to resume from.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


async def evaluate_rag(
    settings: Settings,
    goldens: list[EvalGolden],
    metrics: list[BaseMetric] | None = None,
    limit: int | None = None,
    checkpoint_path: Path | None = None,
) -> EvalReport:
    """Run every golden through RAGComponents.aquery(), score with deepeval.

    Deliberately calls aquery() for EVERY golden (including expect_decline
    ones) to observe the guard's real behavior, then splits: goldens the
    guard declined (or that were expected to decline) go to GuardAccuracy
    only — scoring generation quality against a refusal, or against chunks
    the guard itself judged too weak, is meaningless. Only true negatives
    (expected an answer, guard agreed) go to the deepeval metric suite.

    checkpoint_path, when given, makes both phases resumable: generated
    RAG answers and scored metric results are saved incrementally, and
    already-covered goldens are skipped on the next call with the same
    path. A full multi-hour run used to mean starting over from zero after
    any crash (a dropped tunnel, a HAWKI outage) — this bounds the loss to
    whatever wasn't yet checkpointed. None (the default; what tests use)
    disables checkpointing entirely — no filesystem side effects. On a
    successful full run the checkpoint file is deleted so a later run
    starts fresh instead of silently resuming stale cached results.
    """
    if limit is not None:
        goldens = goldens[:limit]

    judge = HAWKIEvalModel(settings)
    metrics = metrics if metrics is not None else default_metrics(judge)

    checkpoint = (
        _load_checkpoint(checkpoint_path)
        if checkpoint_path
        else {"generated": {}, "scored": {}}
    )
    generated = checkpoint["generated"]
    scored = checkpoint["scored"]

    rag = RAGComponents(settings)
    guard = GuardAccuracy()

    # Bounded-concurrency generation: a plain sequential loop is fine at
    # tens of goldens but becomes the bottleneck at hundreds (each aquery()
    # is a full embed + hybrid-search + HAWKI-generation round trip). Reuses
    # rag_eval_max_concurrent so one setting governs total HAWKI load across
    # both this phase and deepeval's own metric-judging concurrency below.
    semaphore = asyncio.Semaphore(settings.rag_eval_max_concurrent)
    checkpoint_lock = asyncio.Lock()
    to_generate = [g for g in goldens if g.input not in generated]
    if checkpoint_path and len(to_generate) < len(goldens):
        logger.info(
            "Resuming from checkpoint: {}/{} goldens already generated",
            len(goldens) - len(to_generate),
            len(goldens),
        )
    done_count = len(goldens) - len(to_generate)

    async def _run_one(golden: EvalGolden) -> None:
        nonlocal done_count
        async with semaphore:
            session_id = f"eval-{uuid.uuid4().hex[:8]}"
            result = await rag.aquery(golden.input, session_id=session_id)
        async with checkpoint_lock:
            generated[golden.input] = {
                "answer": result.answer,
                "declined": bool(result.trace.get("declined", False)),
                "retrieved_texts": result.trace.get("retrieved_texts", []),
            }
            done_count += 1
            if checkpoint_path and (
                done_count % _GENERATE_CHECKPOINT_EVERY == 0
                or done_count == len(goldens)
            ):
                logger.info("Generated {}/{} goldens", done_count, len(goldens))
                _save_checkpoint(checkpoint_path, checkpoint)

    try:
        if to_generate:
            await asyncio.gather(*(_run_one(g) for g in to_generate))
    finally:
        await rag.aclose()
    if checkpoint_path:
        _save_checkpoint(checkpoint_path, checkpoint)

    test_cases: list[LLMTestCase] = []
    case_meta: list[EvalGolden] = []
    golden_results: list[GoldenResult] = []
    metric_totals: dict[str, list[float]] = {}
    metric_passes: dict[str, list[bool]] = {}

    def _record(golden: EvalGolden, result_data: dict[str, Any]) -> None:
        for name, score in result_data.get("scores", {}).items():
            metric_totals.setdefault(name, []).append(score)
        for name, passed in result_data.get("passes", {}).items():
            metric_passes.setdefault(name, []).append(passed)
        golden_results.append(
            GoldenResult(
                input=golden.input,
                provenance=golden.provenance,
                success=result_data["success"],
                scores=result_data.get("scores", {}),
                reasons=result_data.get("reasons", {}),
                passes=result_data.get("passes", {}),
            )
        )

    for golden in goldens:
        gen = generated[golden.input]
        declined = bool(gen["declined"])

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

        if golden.input in scored:
            _record(golden, scored[golden.input])
            continue

        # LLMTestCase's retrieval_context is list[str | RetrievedContextData]
        # | None; list invariance makes our plain list[str] a structural
        # mismatch under mypy even though it's exactly what deepeval expects
        # at runtime.
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=gen["answer"],
                retrieval_context=cast(Any, gen["retrieved_texts"]),
                expected_output=golden.expected_output,
            )
        )
        case_meta.append(golden)

    if test_cases:
        # deepeval's evaluate() is a sync function that internally drives its
        # own asyncio event loop (loop.run_until_complete(...)). Calling it
        # directly from here — already inside evaluate_rag's running loop —
        # forces deepeval down its nest_asyncio-patched reentrant-loop path.
        # That doesn't raise, but under real load (828 test cases in
        # practice) it silently completed only ~30 before returning, with no
        # error at all — the rest vanished from test_results with nothing in
        # the log to explain it. Running it in a plain thread via
        # run_in_executor gives it a fresh thread with no event loop of its
        # own running, so it takes the normal non-nested code path instead.
        loop = asyncio.get_running_loop()
        batches = [
            (
                case_meta[i : i + _JUDGE_BATCH_SIZE],
                test_cases[i : i + _JUDGE_BATCH_SIZE],
            )
            for i in range(0, len(test_cases), _JUDGE_BATCH_SIZE)
        ]
        judged_so_far = 0
        for batch_num, (batch_meta, batch_cases) in enumerate(batches, start=1):
            eval_result = await loop.run_in_executor(
                None,
                functools.partial(
                    run_deepeval,
                    test_cases=batch_cases,
                    metrics=metrics,
                    async_config=AsyncConfig(
                        max_concurrent=settings.rag_eval_max_concurrent
                    ),
                    display_config=DisplayConfig(
                        print_results=False,
                        show_indicator=False,
                        inspect_after_run=False,
                    ),
                    error_config=ErrorConfig(
                        ignore_errors=True, skip_on_missing_params=True
                    ),
                ),
            )
            for golden, test_result in zip(batch_meta, eval_result.test_results):
                scores: dict[str, float] = {}
                reasons: dict[str, str] = {}
                passes: dict[str, bool] = {}
                for md in test_result.metrics_data or []:
                    if md.score is not None:
                        scores[md.name] = md.score
                        passes[md.name] = md.success
                    if md.reason:
                        reasons[md.name] = md.reason
                result_data = {
                    "success": test_result.success,
                    "scores": scores,
                    "reasons": reasons,
                    "passes": passes,
                }
                _record(golden, result_data)
                if checkpoint_path:
                    scored[golden.input] = result_data

            judged_so_far += len(batch_cases)
            logger.info(
                "Judged batch {}/{} ({} test cases) — {}/{} scored so far",
                batch_num,
                len(batches),
                len(batch_cases),
                judged_so_far,
                len(test_cases),
            )
            if checkpoint_path:
                _save_checkpoint(checkpoint_path, checkpoint)

    summaries = [
        MetricSummary(
            name=name,
            mean_score=sum(scores) / len(scores),
            pass_rate=sum(metric_passes[name]) / len(metric_passes[name]),
            count=len(scores),
        )
        for name, scores in metric_totals.items()
    ]

    report = EvalReport(
        timestamp=datetime.now(UTC).isoformat(),
        golden_count=len(goldens),
        judge_model=judge.get_model_name(),
        guard_accuracy=guard,
        metric_summaries=summaries,
        golden_results=golden_results,
    )

    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()

    return report


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
    # Preferred display order, then any other provenance tags encountered
    # (so a future new source shows up instead of silently vanishing).
    seen = {r.provenance for r in report.golden_results}
    ordered = [p for p in ("anchor", "synthesized", "bibb") if p in seen]
    ordered += sorted(seen - set(ordered))
    for provenance in ordered:
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
    run.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any existing checkpoint and start over from scratch",
    )

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
    setup_logging(settings.log_level, service_name="eval")

    if args.cmd == "run":
        goldens = load_goldens()
        if args.fresh and _DEFAULT_CHECKPOINT_PATH.exists():
            _DEFAULT_CHECKPOINT_PATH.unlink()
            logger.info("Discarded existing checkpoint (--fresh)")
        report = asyncio.run(
            evaluate_rag(
                settings,
                goldens,
                limit=args.limit,
                checkpoint_path=_DEFAULT_CHECKPOINT_PATH,
            )
        )
        print_report(report)
        path = save_report(report, settings)
        logger.info("Report saved: {}", path)
    elif args.cmd == "synthesize":
        from archival_rag.retrieval.search import HybridSearcher
        from scripts.evaluation.dataset import synthesize

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
