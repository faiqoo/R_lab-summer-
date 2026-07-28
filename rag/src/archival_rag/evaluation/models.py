"""Typed data shapes for the evaluation harness.

Mirrors the top-level models.py convention (typed dataclasses over loose
dicts). Stored EvalReport JSON is built from these, not from deepeval's own
result objects directly — insulates historical reports from deepeval's
internal schema shifting across releases.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalGolden:
    """One evaluation question, decoupled from deepeval's own Golden type.

    expect_decline=True marks a deliberately out-of-scope question: it is
    routed to the GuardAccuracy confusion matrix instead of the deepeval RAG
    metrics (an empty/near-miss retrieval_context is meaningless to score).
    """

    input: str
    expected_output: str | None = None
    context: list[str] = field(default_factory=list)
    expect_decline: bool = False
    provenance: str = "anchor"  # "anchor" | "synthesized"
    name: str | None = None


@dataclass
class GuardAccuracy:
    """Confusion matrix for the relevance guard vs. expect_decline goldens."""

    true_positive: int = 0  # expected decline, got decline
    false_negative: int = 0  # expected decline, got answered (guard missed)
    false_positive: int = 0  # expected answer, got decline (guard over-fires)
    true_negative: int = 0  # expected answer, got answered

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_negative
            + self.false_positive
            + self.true_negative
        )

    @property
    def accuracy(self) -> float | None:
        if self.total == 0:
            return None
        return (self.true_positive + self.true_negative) / self.total

    @property
    def precision(self) -> float | None:
        denom = self.true_positive + self.false_positive
        return None if denom == 0 else self.true_positive / denom

    @property
    def recall(self) -> float | None:
        denom = self.true_positive + self.false_negative
        return None if denom == 0 else self.true_positive / denom


@dataclass
class GoldenResult:
    """One golden's outcome: raw per-metric scores, keyed by metric name."""

    input: str
    provenance: str
    success: bool
    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    name: str
    mean_score: float
    pass_rate: float
    count: int


@dataclass
class EvalReport:
    timestamp: str
    golden_count: int
    judge_model: str
    guard_accuracy: GuardAccuracy
    metric_summaries: list[MetricSummary] = field(default_factory=list)
    golden_results: list[GoldenResult] = field(default_factory=list)

    def summary_by(self, provenance: str) -> list[GoldenResult]:
        return [r for r in self.golden_results if r.provenance == provenance]
