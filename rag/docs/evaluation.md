# RAG Evaluation Harness (deepeval)

Systematic quality measurement for the RAG chain, replacing ad hoc manual spot
checks: a golden test corpus (representative questions with known-good
answers/context) scored with [deepeval](https://deepeval.com) RAG metrics.
**Every judge/generation call goes through HAWKI — never OpenAI or any other
external provider** (data governance: archival data and queries stay on
EU/university infrastructure). This is enforced by `HAWKIEvalModel`
(`scripts/evaluation/judge.py`), which every metric and the golden synthesizer use.

## What gets measured

| Metric | Question it answers |
| --- | --- |
| `AnswerRelevancyMetric` | Does the answer actually address the question? |
| `FaithfulnessMetric` | Is the answer grounded in the retrieved chunks (no hallucination)? |
| `ContextualPrecisionMetric` | Are the most relevant retrieved chunks ranked highest? |
| `ContextualRecallMetric` | Did retrieval fetch everything needed to answer? |
| `ContextualRelevancyMetric` | Are the retrieved chunks actually relevant to the question? |

Plus **guard accuracy** — a confusion matrix (TP/FP/FN/TN) comparing each
golden's `expect_decline` flag against whether the relevance guard
(`rag_min_score`) actually declined the question. This directly formalizes
the kind of miscalibration already found once live (`rag_min_score` raised
0.4 → 0.6 after an out-of-scope question slipped through).

## Prerequisites (once)

1. `cd rag && make install-eval` — installs `deepeval` (kept out of the
   default `make install` on purpose: it pulls pandas/rich/tabulate, eval-only
   weight, not needed to run or test the core service).
2. `HAWKI_TOKEN` set in `rag/.env` (same token used everywhere else).
3. For a full run against the live corpus: uni VPN + tunnel, same as
   [gpu-vm-services.md](gpu-vm-services.md) (`make tunnel`, then
   `QDRANT_PORT=16333` / `EMBEDDING_API_URL=...` in `.env`). A quick anchor-only
   smoke run works fine against a local Qdrant too.

## Every work session

```bash
# Cheap smoke check first (anchors only, 2 metrics worth of judge calls):
make eval ARGS="--limit 5"

# Full run (target: 500+ goldens, all 5 metrics — see timing note below):
make eval

# Compare against a previous run (human-eyeball diff, not a pass/fail gate —
# LLM-judge scores are non-deterministic):
make eval-compare BASELINE=eval_runs/2026-07-20T10-00-00.json CANDIDATE=eval_runs/2026-07-28T09-00-00.json
```

**Timing at the ~1000-golden scale**: RAG generation runs with
`rag_eval_max_concurrent` (default 4) concurrent `aquery()` calls — roughly
20-25 minutes for the full dataset. Then deepeval's own metric-judging phase
runs at the same concurrency (5 metrics × 10-20+ judge calls each ≈
thousands of HAWKI calls) — expect this phase alone to take from tens of
minutes to a few hours depending on HAWKI's load. Run it as a
background/overnight job for the full dataset; use `--limit N` for anything
you need back quickly.

**Progress and crash recovery**: `run` logs progress every 50 goldens during
generation and after every batch of 50 during judging (loguru, same stderr +
dated-file logging as everything else — see `rag/logs/`), since deepeval's
own progress indicator is disabled (it doesn't play well with a redirected/
`nohup`'d log). Progress is also checkpointed incrementally to
`eval_runs/.checkpoint.json` — if the process crashes (a dropped tunnel, a
HAWKI outage) partway through, just re-run `make eval`; it resumes from the
checkpoint instead of starting over, skipping already-generated answers and
already-scored metrics. The checkpoint is deleted automatically on a
successful full run. Use `make eval ARGS="--fresh"` to discard an existing
checkpoint and start over deliberately.

Reports are written to `rag/eval_runs/<timestamp>.json` (gitignored — these
are run artifacts, not source). Console output breaks scores down by
**anchor** (hand-curated, human-reviewed — the trustworthy signal),
**synthesized** (LLM-generated from real chunks — broader coverage, noisier),
and **bibb** (below) separately. Don't average them together.

## The golden dataset

- `tests/eval_data/goldens_anchor.json` — hand-curated, checked in, edit by
  hand + review in PR. ~10 questions from documents already manually verified
  (Bäcker 1983/2004, DDR-era Drechsler, Arzthelfer 1985) plus deliberate
  out-of-scope questions (`expect_decline: true`) like "Hauptstadt von
  Frankreich".
- `tests/eval_data/goldens_synthesized.json` — regenerated on demand:

  ```bash
  make eval-dataset
  ```

  This pulls real indexed chunk text from the live Qdrant collection, groups
  it by source document, and has `HAWKIEvalModel` (via deepeval's
  `Synthesizer`) write plausible questions grounded in that real text.
  **Review the output before committing** — synthesized `expected_output` is
  LLM-generated and can be wrong.

  Default: 200 distinct source documents × up to 3 questions each = up to
  600 goldens generated (deepeval's own quality filtration typically drops
  some fraction), targeting comfortably over the 500-golden minimum after
  filtering, spread across whatever occupations/eras those 200 documents
  cover — `_sample_contexts` paginates the live collection until it finds
  that many distinct documents (not just a single fixed-size page), so the
  actual spread depends on the collection, not on scroll ordering luck.
  `--n-groups` is the real breadth lever (more *distinct documents*
  represented), not `--max-goldens-per-context` (more repetitive questions
  about the same handful of documents):

  ```bash
  # Go wider than the 500-target default:
  uv run --group eval --env-file .env python -m scripts.evaluation.harness \
    synthesize --n-groups 400 --chunks-per-group 4
  ```

- `tests/eval_data/goldens_bibb.json` — 106 goldens instantiated from a real
  question/occupation handoff BIBB (the archive's supervising body) gave
  Pravina in 2026 (`2026-01-29 Fragen Chatbot_ Rückfragen TR.xlsx` +
  `2026-03-02 Berufe Chatbot bereinigt.xlsx`). BIBB supplied template
  questions with placeholders (`BERUFSBEZEICHNUNG`, `ERSCHEINUNGSJAHR` /
  `ERLASSDATUM` / `INKRAFTTRETENSDATUM`) plus a separate table of real
  documents to fill them with. Only templates BIBB's own questions were
  actually answerable by *this* system were instantiated:
  - Documents-from-year(s) X — the placeholder is filled using the same
    Erlassdatum > Inkrafttreten > Erscheinungsjahr precedence BIBB specified
    (see "Year extraction precedence" below), including dual-year variants
    ("... aus den Jahren 1980 und 1982?") for occupations with 2+ dated
    editions.
  - Training content / training duration per occupation — `expected_output`
    left unset (BIBB's own template had no filled example to build ground
    truth from); graded on answer relevancy/faithfulness against live
    retrieval instead of exact match.
  - "What is an Anlernberuf?" — a single ready-made golden, no placeholder.

  Only occupations from BIBB's list confirmed present in the live Qdrant
  collection were used (33 of 52 named occupations matched at generation
  time — the rest either aren't ingested yet or are named differently).

  **Deliberately excluded, not just deprioritized:**
  - BIBB's own `zurückgestellt` ("postponed") sheet, and template #6 (DDR
    occupation-name mapping) — both explicitly need Genealogien/
    Entsprechungslisten data we don't have.
  - Templates #8/#11 (abolition date) — no `Aufhebungsdatum` field exists in
    BIBB's document table, and #8 had no filled reference example.
  - **Templates #1, #13, #14, #15 — all ask for an exact document *count***
    ("how many documents do you have for X", "how many in total", "how many
    digitally"). This is a real capability gap, not a data gap: `/chat` only
    ever retrieves the top-`k` chunks and generates from them — it has no
    path to count matching documents, and never will unless a separate
    structured-query endpoint is added. Writing goldens for these would test
    a feature that doesn't exist.

## Year extraction precedence

BIBB's stated rule when a document carries multiple dates: **Erlassdatum
(decree date) > Inkrafttreten (effective date) > Erscheinungsjahr/Stand
(publication year)**. `_extract_year()` in `ingestion/ingest.py` implements
exactly this — first an explicit "vom `<date>`" decree phrase, then "tritt
... in Kraft" / "in Kraft getreten am `<date>`", then any bare 4-digit year
as a last resort. This is what makes goldens like "Habt ihr etwas zum Beruf
Maurer aus dem Jahr 1985?" answerable with the *correct* year rather than
whichever date happens to appear first in the text.

Each golden × each of the 5 metrics can fan out into 10-20+ underlying judge
calls (faithfulness does claim-extraction + per-claim verification;
contextual-precision does per-chunk verdicts) — at 500+ goldens that's
thousands of HAWKI calls per full run. `rag_eval_max_concurrent` (default 4)
throttles this and governs generation, synthesis, and metric-judging
concurrency together; raise it cautiously and only after confirming HAWKI
holds up under the higher load.

## Judge model choice

`rag_eval_model` (default `openai-gpt-oss-120b`, HAWKI-hosted — no external
OpenAI calls despite the name) is deliberately **not** the same as
`hawki_model` (the model that generates chat answers): grading an answer with
the same model that produced it biases scores upward. If you change this,
sanity-check that the model reliably emits clean structured JSON — HAWKI
models vary here (`ChatHAWKI.with_structured_output` already has a
fence-stripping fallback for exactly this reason), and a judge that can't
follow the schema will show up as spurious metric errors, not a quality
signal.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: deepeval` | Run `make install-eval` first |
| `ValueError: HAWKI_TOKEN is required` | Set `HAWKI_TOKEN` in `rag/.env` |
| Run takes a long time / times out | Lower `--limit`, or check HAWKI isn't under load — this is a shared endpoint |
| Metric scores all `None`/errors | `error_config` swallows per-metric failures (`skip_on_missing_params`) rather than aborting the batch — check the `reason` field in the saved report JSON for the actual cause |
| `make eval-dataset` produces 0 goldens | Qdrant collection is empty or unreachable — check the tunnel (`docs/gpu-vm-services.md`) |
| Run crashed (tunnel drop, HAWKI outage) | Just re-run `make eval` — it resumes from `eval_runs/.checkpoint.json` automatically |
| Want to discard a stale checkpoint and start clean | `make eval ARGS="--fresh"`, or just delete `eval_runs/.checkpoint.json` |

## Rules

- Never point the judge or synthesizer at anything other than `HAWKIEvalModel`
  — no OpenAI keys, no external providers.
- Treat `make eval-compare` as a discussion starter, not a CI gate: LLM-judge
  scores are inherently noisy run-to-run.
- Don't commit `rag/eval_runs/` (gitignored) — only the goldens under
  `tests/eval_data/` are source of truth.
