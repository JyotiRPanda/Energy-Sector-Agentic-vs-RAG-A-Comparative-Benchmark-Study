from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Citation:
    source_file: str
    table_id: str | None = None
    row_id: str | None = None
    column_id: str | None = None
    evidence_text: str | None = None
    pdf_name: str | None = None
    page_nbr: str | None = None
    table_nbr: str | None = None
    primary_value: str | None = None
    evidence_id: str | None = None
    reason_used: str | None = None


@dataclass
class BenchmarkExample:
    question_id: str
    question: str
    gold_answer: str
    split: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    question_id: str
    pipeline_name: str
    answer: str
    latency_ms: float
    citations: list[Citation] = field(default_factory=list)
    trace_steps: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredPrediction:
    prediction: Prediction
    exact_match: float
    numeric_relative_error: float | None
    citation_precision: float | None
    citation_recall: float | None
    faithfulness_score: float
    transparency_score: float
    error_labels: list[str] = field(default_factory=list)
