"""Complex-subset evaluation for quantitative, multi-step, and multi-table questions.

Provides focused metrics on the question types where agentic reasoning should excel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gri_benchmark.types import BenchmarkExample, Prediction, ScoredPrediction
from gri_benchmark.evaluation.metrics import (
    exact_match,
    numeric_relative_error,
    citation_precision,
    citation_recall,
    faithfulness_score,
    transparency_score,
)
from gri_benchmark.evaluation.error_taxonomy import classify_errors


def score_predictions(
    examples: list[BenchmarkExample],
    predictions: list[Prediction],
) -> list[ScoredPrediction]:
    """Convert predictions into scored predictions."""
    ex_by_id = {ex.question_id: ex for ex in examples}
    scored = []
    
    for pred in predictions:
        if pred.question_id not in ex_by_id:
            continue
        
        ex = ex_by_id[pred.question_id]
        
        em = exact_match(ex.gold_answer, pred.answer)
        nre = numeric_relative_error(ex.gold_answer, pred.answer)
        cp = citation_precision(ex, pred)
        cr = citation_recall(pred, ex)
        faith = faithfulness_score(ex, pred)
        trans = transparency_score(pred)
        error_labels = classify_errors(ex, pred)
        
        scored.append(
            ScoredPrediction(
                prediction=pred,
                exact_match=em,
                numeric_relative_error=nre,
                citation_precision=cp,
                citation_recall=cr,
                faithfulness_score=faith,
                transparency_score=trans,
                error_labels=error_labels,
            )
        )
    
    return scored


@dataclass
class ComplexSubsetMetrics:
    """Metrics for a subset of complex questions."""
    subset_name: str
    question_count: int
    exact_match: float
    numeric_relative_error: float | None
    citation_precision: float | None
    faithfulness_score: float
    transparency_score: float
    error_labels: dict[str, int] = field(default_factory=dict)
    questions_by_type: dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0


def get_subset_by_split(
    examples: list[BenchmarkExample],
    scored_predictions: list[ScoredPrediction],
    split_prefix: str,
) -> tuple[list[BenchmarkExample], list[ScoredPrediction]]:
    """Extract subset of examples/predictions matching a split prefix."""
    subset_examples = []
    subset_predictions = []
    
    example_dict = {ex.question_id: ex for ex in examples}
    
    for pred in scored_predictions:
        if pred.prediction.question_id not in example_dict:
            continue
        
        example = example_dict[pred.prediction.question_id]
        if example.split.startswith(split_prefix):
            subset_examples.append(example)
            subset_predictions.append(pred)
    
    return subset_examples, subset_predictions


def compute_subset_metrics(
    examples: list[BenchmarkExample],
    scored_predictions: list[ScoredPrediction],
    subset_name: str,
    split_prefix: str | None = None,
    question_types: list[str] | None = None,
) -> ComplexSubsetMetrics:
    """Compute metrics for a subset of questions.
    
    Args:
        examples: Full list of benchmark examples
        scored_predictions: Full list of scored predictions
        subset_name: Name of this subset for reporting
        split_prefix: If provided, filter to splits starting with this prefix
        question_types: If provided, filter to these question types
    """
    # Filter to subset
    subset_preds = scored_predictions
    subset_examples = examples
    
    if split_prefix:
        subset_examples, subset_preds = get_subset_by_split(examples, scored_predictions, split_prefix)
    
    if not subset_preds:
        return ComplexSubsetMetrics(
            subset_name=subset_name,
            question_count=0,
            exact_match=0.0,
            numeric_relative_error=None,
            citation_precision=None,
            faithfulness_score=0.0,
            transparency_score=0.0,
        )
    
    # Aggregate metrics
    em_sum = sum(p.exact_match for p in subset_preds)
    nre_values = [p.numeric_relative_error for p in subset_preds if p.numeric_relative_error is not None]
    cp_values = [p.citation_precision for p in subset_preds if p.citation_precision is not None]
    faith_sum = sum(p.faithfulness_score for p in subset_preds)
    trans_sum = sum(p.transparency_score for p in subset_preds)
    
    # Error labels
    error_counts: dict[str, int] = {}
    for pred in subset_preds:
        for label in pred.error_labels:
            error_counts[label] = error_counts.get(label, 0) + 1
    
    # Latency and cost
    latency_values = [p.prediction.latency_ms for p in subset_preds]
    cost_values = [p.prediction.metadata.get("cost_usd", 0.0) for p in subset_preds]
    
    # Question types
    questions_by_type = {}
    for ex in subset_examples:
        qtype = ex.split.split("_")[-1] if "_" in ex.split else "unknown"
        questions_by_type[qtype] = questions_by_type.get(qtype, 0) + 1
    
    return ComplexSubsetMetrics(
        subset_name=subset_name,
        question_count=len(subset_preds),
        exact_match=em_sum / len(subset_preds) if subset_preds else 0.0,
        numeric_relative_error=sum(nre_values) / len(nre_values) if nre_values else None,
        citation_precision=sum(cp_values) / len(cp_values) if cp_values else None,
        faithfulness_score=faith_sum / len(subset_preds) if subset_preds else 0.0,
        transparency_score=trans_sum / len(subset_preds) if subset_preds else 0.0,
        error_labels=error_counts,
        questions_by_type=questions_by_type,
        avg_latency_ms=sum(latency_values) / len(latency_values) if latency_values else 0.0,
        avg_cost_usd=sum(cost_values) / len(cost_values) if cost_values else 0.0,
    )


def compute_all_complex_subsets(
    examples: list[BenchmarkExample],
    scored_predictions: list[ScoredPrediction],
) -> dict[str, ComplexSubsetMetrics]:
    """Compute metrics for all important complex subsets."""
    subsets: dict[str, ComplexSubsetMetrics] = {}
    
    # By question type
    subsets["quantitative_questions"] = compute_subset_metrics(
        examples, scored_predictions,
        "Quantitative Questions",
        split_prefix="single_table_quantitative"
    )
    
    subsets["multi_table_questions"] = compute_subset_metrics(
        examples, scored_predictions,
        "Multi-Table Questions",
        split_prefix="multi_table"
    )
    
    subsets["multistep_questions"] = compute_subset_metrics(
        examples, scored_predictions,
        "Multi-Step Questions",
        split_prefix="multi_table_multistep"
    )
    
    subsets["relational_questions"] = compute_subset_metrics(
        examples, scored_predictions,
        "Relational Questions",
        split_prefix="single_table_relational"
    )
    
    subsets["extractive_questions"] = compute_subset_metrics(
        examples, scored_predictions,
        "Extractive Questions",
        split_prefix="single_table_extractive"
    )
    
    # Combined complex
    multi_subset_preds = []
    for pred in scored_predictions:
        if (pred.prediction.question_id in {ex.question_id for ex in examples 
            if ex.split.startswith(("multi_table", "single_table_quantitative", "single_table_relational"))}):
            multi_subset_preds.append(pred)
    
    subsets["all_complex_questions"] = compute_subset_metrics(
        examples, scored_predictions,
        "All Complex Questions (Multi-table + Quantitative + Relational)"
    )
    
    return subsets


def format_complex_subset_report(subsets: dict[str, ComplexSubsetMetrics]) -> str:
    """Format complex subset metrics as markdown."""
    lines = ["# Complex Subset Evaluation\n"]
    
    for subset_name, metrics in subsets.items():
        lines.append(f"## {metrics.subset_name}\n")
        lines.append(f"**Question Count:** {metrics.question_count}\n")
        lines.append(f"**Exact Match:** {metrics.exact_match:.4f}\n")
        
        if metrics.numeric_relative_error is not None:
            lines.append(f"**Numeric Relative Error:** {metrics.numeric_relative_error:.4f}\n")
        
        if metrics.citation_precision is not None:
            lines.append(f"**Citation Precision:** {metrics.citation_precision:.4f}\n")
        
        lines.append(f"**Faithfulness:** {metrics.faithfulness_score:.4f}\n")
        lines.append(f"**Transparency:** {metrics.transparency_score:.4f}\n")
        lines.append(f"**Avg Latency (ms):** {metrics.avg_latency_ms:.2f}\n")
        lines.append(f"**Avg Cost (USD):** ${metrics.avg_cost_usd:.6f}\n")
        
        if metrics.questions_by_type:
            lines.append("**Question Type Breakdown:**\n")
            for qtype, count in sorted(metrics.questions_by_type.items()):
                lines.append(f"  - {qtype}: {count}\n")
        
        if metrics.error_labels:
            lines.append("**Top Error Labels:**\n")
            sorted_errors = sorted(metrics.error_labels.items(), key=lambda x: x[1], reverse=True)
            for label, count in sorted_errors[:5]:
                lines.append(f"  - {label}: {count}\n")
        
        lines.append("\n")
    
    return "".join(lines)


def compare_pipeline_subsets(
    pipeline1_subsets: dict[str, ComplexSubsetMetrics],
    pipeline2_subsets: dict[str, ComplexSubsetMetrics],
    pipeline1_name: str,
    pipeline2_name: str,
) -> str:
    """Compare complex subset metrics between two pipelines."""
    lines = [f"# Complex Subset Comparison: {pipeline1_name} vs {pipeline2_name}\n\n"]
    
    for subset_name in sorted(pipeline1_subsets.keys()):
        if subset_name not in pipeline2_subsets:
            continue
        
        m1 = pipeline1_subsets[subset_name]
        m2 = pipeline2_subsets[subset_name]
        
        lines.append(f"## {m1.subset_name}\n")
        
        # Exact match comparison
        em_diff = m2.exact_match - m1.exact_match
        em_pct = (em_diff / m1.exact_match * 100) if m1.exact_match > 0 else 0
        lines.append(f"**Exact Match:** {m1.exact_match:.4f} → {m2.exact_match:.4f} ({em_pct:+.1f}%)\n")
        
        # Numeric relative error if available
        if m1.numeric_relative_error is not None and m2.numeric_relative_error is not None:
            nre_diff = m2.numeric_relative_error - m1.numeric_relative_error
            nre_pct = (nre_diff / m1.numeric_relative_error * 100) if m1.numeric_relative_error > 0 else 0
            lines.append(f"**Numeric Relative Error:** {m1.numeric_relative_error:.4f} → {m2.numeric_relative_error:.4f} ({nre_pct:+.1f}%)\n")
        
        # Citation precision if available
        if m1.citation_precision is not None and m2.citation_precision is not None:
            cp_diff = m2.citation_precision - m1.citation_precision
            cp_pct = (cp_diff / m1.citation_precision * 100) if m1.citation_precision > 0 else 0
            lines.append(f"**Citation Precision:** {m1.citation_precision:.4f} → {m2.citation_precision:.4f} ({cp_pct:+.1f}%)\n")
        
        # Latency comparison
        latency_diff = m2.avg_latency_ms - m1.avg_latency_ms
        lines.append(f"**Avg Latency (ms):** {m1.avg_latency_ms:.2f} → {m2.avg_latency_ms:.2f} ({latency_diff:+.2f})\n")
        
        lines.append("\n")
    
    return "".join(lines)
