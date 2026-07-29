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
