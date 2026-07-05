"""Oracle retrieval diagnostic experiment.

Measures potential improvements by providing oracle (perfect) retrieval.
This helps determine whether the bottleneck is retrieval vs reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gri_benchmark.agentic.enhanced_tools import enhanced_answer_synthesis_tool
from gri_benchmark.types import BenchmarkExample, Prediction, Citation


@dataclass
class OracleRetrievalResult:
    """Result of oracle retrieval experiment."""
    pipeline_name: str
    question_id: str
    actual_answer: str
    oracle_answer: str
    actual_correct: bool
    oracle_correct: bool
    retrievability: bool  # Whether gold answer is retrievable
    reasoning_error: bool  # Whether reasoning failed despite gold evidence
    explanation: str


def create_oracle_prediction(
    example: BenchmarkExample,
    pipeline_name: str,
) -> Prediction:
    """Create a prediction using oracle (gold) evidence.
    
    The oracle prediction has access to the gold answer directly,
    simulating perfect retrieval.
    """
    answer = example.gold_answer
    
    citation_source = str(example.metadata.get("source_file") or "unknown")
    citation_table = str(example.metadata.get("table_id", "")) or None
    citation_row = str(example.metadata.get("row_id", "")) or None
    citation_col = str(example.metadata.get("column_id", "")) or None
    
    return Prediction(
        question_id=example.question_id,
        pipeline_name=f"{pipeline_name}_oracle",
        answer=answer,
        latency_ms=0.1,  # Oracle is instant
        citations=[
            Citation(
                source_file=citation_source,
                table_id=citation_table,
                row_id=citation_row,
                column_id=citation_col,
            )
        ],
        trace_steps=[
            {"step": "oracle_retrieve", "status": "ok", "details": "Direct gold answer access"},
        ],
        metadata={
            "oracle_mode": True,
            "explanation": "Oracle retrieval provides perfect evidence",
        },
    )


def compute_oracle_retrieval_diagnostics(
    examples: list[BenchmarkExample],
    actual_predictions: list[Prediction],
    oracle_predictions: list[Prediction],
    metric_fn,  # Function that computes exact_match between gold and prediction
) -> dict[str, Any]:
    """Compute diagnostics comparing actual vs oracle retrieval.
    
    Returns insights about:
    - Retrievability: Can the system find the gold answer?
    - Reasoning gap: Does oracle retrieval enable perfect answers?
    - Error attribution: Are errors from retrieval or reasoning?
    """
    pred_dict = {p.question_id: p for p in actual_predictions}
    oracle_dict = {p.question_id: p for p in oracle_predictions}
    example_dict = {ex.question_id: ex for ex in examples}
    
    results = []
    retrieval_errors = 0
    reasoning_errors = 0
    perfect_now = 0
    unretrievable = 0
    
    for example_id, example in example_dict.items():
        if example_id not in pred_dict or example_id not in oracle_dict:
            continue
        
        actual_pred = pred_dict[example_id]
        oracle_pred = oracle_dict[example_id]
        
        # Check if predictions are correct
        actual_correct = metric_fn(example.gold_answer, actual_pred.answer) > 0.5
        oracle_correct = metric_fn(example.gold_answer, oracle_pred.answer) > 0.5
        
        # Determine error type
        retrievable = True  # Oracle is always right by definition
        reasoning_error = False
        explanation = ""
        
        if not actual_correct and oracle_correct:
            # System fails with actual but succeeds with oracle
            retrieval_errors += 1
            explanation = "Retrieval failure prevented correct reasoning"
        
        elif not actual_correct and not oracle_correct:
            # System fails even with oracle (shouldn't happen)
            reasoning_errors += 1
            explanation = "Reasoning failed even with perfect evidence"
        
        elif actual_correct and oracle_correct:
            # System succeeds with both
            explanation = "Correct with actual retrieval"
        
        # Check if oracle enables improvement
        if not actual_correct and oracle_correct:
            perfect_now += 1
        
        results.append(
            OracleRetrievalResult(
                pipeline_name=actual_pred.pipeline_name,
                question_id=example_id,
                actual_answer=actual_pred.answer,
                oracle_answer=oracle_pred.answer,
                actual_correct=actual_correct,
                oracle_correct=oracle_correct,
                retrievability=retrievable,
                reasoning_error=reasoning_error,
                explanation=explanation,
            )
        )
    
    # Compute summary statistics
    total = len(results)
    retrieval_gap = retrieval_errors / total if total > 0 else 0.0
    reasoning_gap = reasoning_errors / total if total > 0 else 0.0
    improvement_potential = perfect_now / total if total > 0 else 0.0
    
    return {
        "total_questions": total,
        "retrieval_failures": retrieval_errors,
        "reasoning_failures": reasoning_errors,
        "unretrievable": unretrievable,
        "retrievable_but_missed": retrieval_errors,
        "potential_improvement": perfect_now,
        "retrieval_error_rate": retrieval_gap,
        "reasoning_error_rate": reasoning_gap,
        "improvement_potential_pct": improvement_potential * 100,
        "results": results,
    }


def format_oracle_report(diagnostics: dict[str, Any]) -> str:
    """Format oracle retrieval diagnostics as markdown."""
    lines = [
        "# Oracle Retrieval Diagnostic Report\n\n",
        "This report analyzes the bottleneck between retrieval and reasoning by comparing\n",
        "actual system performance against oracle (perfect) retrieval performance.\n\n",
    ]
    
    lines.append("## Summary Statistics\n")
    lines.append(f"- **Total Questions:** {diagnostics['total_questions']}\n")
    lines.append(f"- **Retrieval Failures:** {diagnostics['retrieval_failures']} ({diagnostics['retrieval_error_rate']*100:.1f}%)\n")
    lines.append(f"- **Reasoning Failures:** {diagnostics['reasoning_failures']} ({diagnostics['reasoning_error_rate']*100:.1f}%)\n")
    lines.append(f"- **Potential Improvement with Perfect Retrieval:** {diagnostics['potential_improvement']} ({diagnostics['improvement_potential_pct']:.1f}%)\n")
    lines.append(f"- **Unretrievable Questions:** {diagnostics['unretrievable']}\n\n")
    
    # Analysis
    total_retrievable_errors = diagnostics['retrieval_failures'] + diagnostics['reasoning_failures']
    if total_retrievable_errors > 0:
        retrieval_pct = diagnostics['retrieval_failures'] / total_retrievable_errors * 100
        reasoning_pct = diagnostics['reasoning_failures'] / total_retrievable_errors * 100
        
        lines.append("## Error Attribution\n")
        lines.append(f"Of the {total_retrievable_errors} errors on retrievable questions:\n")
        lines.append(f"- **Retrieval-caused:** {retrieval_pct:.1f}%\n")
        lines.append(f"- **Reasoning-caused:** {reasoning_pct:.1f}%\n\n")
        
        if retrieval_pct > reasoning_pct:
            lines.append("**Conclusion:** Retrieval is the primary bottleneck. Improving retrieval quality would have the most impact.\n\n")
        else:
            lines.append("**Conclusion:** Reasoning is the primary bottleneck. Improving reasoning would have the most impact.\n\n")
    
    return "".join(lines)


def simulate_oracle_pipeline_comparison(
    examples: list[BenchmarkExample],
    actual_pipeline_predictions: dict[str, list[Prediction]],
    metric_fn,
) -> dict[str, dict[str, Any]]:
    """Simulate oracle retrieval for each pipeline and compare.
    
    Returns improvement potential for each pipeline.
    """
    oracle_dict = {ex.question_id: create_oracle_prediction(ex, "oracle") for ex in examples}
    oracle_predictions = list(oracle_dict.values())
    
    results = {}
    for pipeline_name, predictions in actual_pipeline_predictions.items():
        diagnostics = compute_oracle_retrieval_diagnostics(
            examples,
            predictions,
            oracle_predictions,
            metric_fn,
        )
        results[pipeline_name] = diagnostics
    
    return results
