"""RAG evaluation harness (deepeval). See docs/evaluation.md.

Not imported by the API or ingestion code — install with
``uv sync --dev --group eval`` (``make install-eval``), separate from the
core runtime dependencies.
"""

from __future__ import annotations

import os

# Known deepeval backward-compat quirk: only the literal "1" reliably disables
# telemetry across versions ("true"/"YES" have been silently ignored in past
# releases — see confident-ai/deepeval issues #2231, #1613). Set here, at
# package import time, so it's opt-out by default without requiring every
# contributor to configure their shell.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

# deepeval wraps the ENTIRE evaluate() batch in asyncio.wait_for(gather(...),
# timeout=DEEPEVAL_PER_TASK_TIMEOUT_SECONDS + buffer) — a single ~207s
# deadline (180s default + ~27s buffer) for the whole run, not per test case.
# It assumes concurrency is high enough that every task finishes around the
# same time; at our concurrency (rag_eval_max_concurrent, typically 4) with
# hundreds of goldens queued behind it, most tasks are still in-flight when
# that fixed deadline hits and get silently cancelled — confirmed live
# 2026-08-21: a 1031-golden run reported only 30 "Test Results", the rest
# vanished with no error, no traceback, nothing — the run duration (208.57s)
# matched the default timeout almost exactly. Disabling deepeval's own
# timeout layer here and relying on HAWKIClientMixin's own retry/backoff
# (llm/client.py — bounded, per-call, already correct) avoids this rather
# than trying to guess a "big enough" override number.
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")
