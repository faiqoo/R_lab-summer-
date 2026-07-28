"""Golden dataset: load/merge hand-curated + synthesized goldens.

Anchor goldens (tests/eval_data/goldens_anchor.json) are hand-curated and
human-reviewed — the trustworthy regression signal. Synthesized goldens
(tests/eval_data/goldens_synthesized.json) are generated from real indexed
chunk text via deepeval's Synthesizer + HAWKIEvalModel — broader coverage,
but noisier (LLM-generated expected_output can itself be wrong). Report
scores for the two separately; never average them into one number.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import asdict
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from loguru import logger

from archival_rag.config import Settings
from archival_rag.evaluation.judge import HAWKIEvalModel
from archival_rag.evaluation.models import EvalGolden
from archival_rag.retrieval.search import HybridSearcher

_DEFAULT_ANCHOR_PATH = (
    Path(__file__).resolve().parents[3] / "tests/eval_data/goldens_anchor.json"
)
_DEFAULT_SYNTH_PATH = (
    Path(__file__).resolve().parents[3] / "tests/eval_data/goldens_synthesized.json"
)


def _load_json_goldens(path: Path, provenance: str) -> list[EvalGolden]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalGolden(
            input=item["input"],
            expected_output=item.get("expected_output"),
            context=item.get("context", []),
            expect_decline=item.get("expect_decline", False),
            provenance=provenance,
            name=item.get("name"),
        )
        for item in raw
    ]


def load_goldens(
    anchor_path: Path = _DEFAULT_ANCHOR_PATH,
    synthesized_path: Path = _DEFAULT_SYNTH_PATH,
) -> list[EvalGolden]:
    """Load and merge the anchor + synthesized golden files.

    A missing anchor file is an error (it's the required, checked-in seed);
    a missing synthesized file is fine (not yet generated via
    ``make eval-dataset``).
    """
    if not anchor_path.exists():
        raise FileNotFoundError(f"Required anchor goldens not found: {anchor_path}")
    anchors = _load_json_goldens(anchor_path, provenance="anchor")
    synthesized = _load_json_goldens(synthesized_path, provenance="synthesized")
    return anchors + synthesized


def save_goldens(goldens: list[EvalGolden], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {k: v for k, v in asdict(g).items() if k != "provenance"} for g in goldens
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# Safety bound on total points scanned while paginating for distinct source
# documents — generous enough to find hundreds of documents in a ~65k-chunk,
# ~5k-document collection without risking an unbounded scan.
_MAX_SCAN_POINTS = 60_000
_SCROLL_PAGE_SIZE = 1000


async def _sample_contexts(
    searcher: HybridSearcher,
    collection: str,
    n_groups: int,
    chunks_per_group: int,
    seed: int = 42,
) -> list[list[str]]:
    """Group real indexed chunk texts by source document into context groups.

    Paginates the live collection (no query bias from hand-picked seed
    terms) until at least n_groups distinct source documents are found, the
    collection is exhausted, or the scan safety bound is hit — a single
    fixed-size scroll can't reliably surface e.g. 200+ distinct documents
    regardless of how Qdrant happens to order the collection internally.
    Groups chunks by their source PDF so each context group reads as one
    coherent document excerpt, and returns a random sample of groups.
    """
    by_source: dict[str, list[str]] = {}
    offset = None
    scanned = 0

    while scanned < _MAX_SCAN_POINTS:
        points, offset = await searcher.async_client.scroll(
            collection_name=collection,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=["text", "source", "chunk_type"],
            with_vectors=False,
        )
        scanned += len(points)
        for p in points:
            payload = p.payload or {}
            text = payload.get("text", "").strip()
            if not text or payload.get("chunk_type") != "text":
                continue
            by_source.setdefault(payload.get("source", ""), []).append(text)

        if offset is None:
            break  # collection exhausted
        if len(by_source) >= n_groups:
            break

    groups = [texts[:chunks_per_group] for texts in by_source.values() if texts]
    random.Random(seed).shuffle(groups)
    logger.info(
        "Sampled {} distinct source documents ({} scanned points) for synthesis",
        len(groups),
        scanned,
    )
    return groups[:n_groups]


# Contexts per synthesizer call. deepeval's generate_goldens_from_contexts()
# has no error_config/ignore_errors equivalent — a single unparseable HAWKI
# response anywhere in the batch raises and loses the ENTIRE batch's work
# (confirmed live 2026-07-28: died ~15 calls into a 200-context run with
# nothing saved). Batching bounds the blast radius of one bad response to a
# handful of goldens instead of the whole run, and lets us save progress
# after every batch instead of only at the very end. Live-tested 2026-07-28:
# at batch size 10 / concurrency 4 sustained over ~18 minutes, HAWKI (a
# shared, non-dedicated university endpoint) hit a 75% batch failure rate in
# bursty clusters, not a steady degradation — smaller batches + more patient
# retries ride out congestion windows better than either alone.
_SYNTHESIS_BATCH_SIZE = 5
_RETRY_BACKOFF_SECONDS = [15, 45, 90]


async def synthesize(
    settings: Settings,
    searcher: HybridSearcher,
    out_path: Path = _DEFAULT_SYNTH_PATH,
    n_groups: int = 200,
    chunks_per_group: int = 4,
    max_goldens_per_context: int = 3,
) -> list[EvalGolden]:
    """Generate goldens from real indexed chunk text via deepeval's Synthesizer.

    Grounds every synthesized question in text that is actually retrievable
    from the live collection, and keeps generation on HAWKI (never OpenAI)
    via HAWKIEvalModel — specifically settings.rag_synthesis_model, not
    rag_eval_model: there's no self-grading bias concern for question
    generation, so this uses whichever model is most reliable under real
    load, independent of the metric-judging model choice.

    n_groups is the real breadth lever — it controls how many distinct
    source documents (occupations/eras) get represented, not just how many
    goldens come out; prefer raising it over max_goldens_per_context if you
    want wider corpus coverage rather than more, repetitive questions about
    the same handful of documents.

    Saves incrementally after every batch, so a failure partway through a
    long run still leaves everything generated up to that point on disk.
    Loads any goldens already at out_path first and appends to them (deduped
    by input text) rather than overwriting — repeated runs accumulate toward
    a target instead of each run discarding the last one's progress.
    """
    contexts = await _sample_contexts(
        searcher, settings.qdrant_collection, n_groups, chunks_per_group
    )
    if not contexts:
        logger.warning("No text chunks found to synthesize goldens from")
        return []

    judge = HAWKIEvalModel(settings, model_name=settings.rag_synthesis_model)
    # async_mode=True + max_concurrent ties synthesis load to the same
    # rag_eval_max_concurrent knob used for RAG generation and metric
    # judging elsewhere in the harness — one setting governs total HAWKI load.
    synthesizer = Synthesizer(
        model=judge, async_mode=True, max_concurrent=settings.rag_eval_max_concurrent
    )

    goldens = _load_json_goldens(out_path, provenance="synthesized")
    seen_inputs = {g.input for g in goldens}
    starting_count = len(goldens)
    failed_batches = 0
    batches = [
        contexts[i : i + _SYNTHESIS_BATCH_SIZE]
        for i in range(0, len(contexts), _SYNTHESIS_BATCH_SIZE)
    ]
    for i, batch in enumerate(batches):
        raw_goldens = None
        # Bounded retry with backoff on top of the batch-level isolation
        # above: HAWKI (a shared, non-dedicated university endpoint) showed
        # bursty congestion windows live on 2026-07-28 — several consecutive
        # batches failing together, then succeeding again minutes later —
        # so a few patient retries ride out a congestion window instead of
        # giving up on the first bad response.
        for attempt, backoff in enumerate([*_RETRY_BACKOFF_SECONDS, None]):
            try:
                raw_goldens = synthesizer.generate_goldens_from_contexts(
                    contexts=batch,
                    include_expected_output=True,
                    max_goldens_per_context=max_goldens_per_context,
                    _send_data=False,
                )
                break
            except Exception:
                if backoff is not None:
                    logger.warning(
                        "Synthesis batch {}/{} failed (attempt {}), " "retrying in {}s",
                        i + 1,
                        len(batches),
                        attempt + 1,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                else:
                    failed_batches += 1
                    logger.exception(
                        "Synthesis batch {}/{} failed after {} attempts "
                        "({} documents) — skipping, continuing with "
                        "remaining batches",
                        i + 1,
                        len(batches),
                        len(_RETRY_BACKOFF_SECONDS) + 1,
                        len(batch),
                    )
        if raw_goldens is None:
            continue

        new_goldens = [
            EvalGolden(
                input=g.input,
                expected_output=g.expected_output,
                context=g.context or [],
                provenance="synthesized",
            )
            for g in raw_goldens
            if g.input and g.input not in seen_inputs
        ]
        seen_inputs.update(g.input for g in new_goldens)
        goldens.extend(new_goldens)
        save_goldens(goldens, out_path)
        logger.info(
            "Batch {}/{} done: {} goldens so far ({} failed batches)",
            i + 1,
            len(batches),
            len(goldens),
            failed_batches,
        )

    logger.info(
        "Synthesized {} new goldens ({} total) -> {} ({}/{} batches failed)",
        len(goldens) - starting_count,
        len(goldens),
        out_path,
        failed_batches,
        len(batches),
    )
    return goldens
