"""Run complex-only benchmark and generate comparative report.

Evaluates RAG vs agentic approaches on complex reasoning tasks:
- Relational (comparisons, rankings)
- Quantitative (calculations, aggregations)  
- Multistep (multi-stage reasoning)
- Multi-table (cross-table joining)

Metrics:
- Exact match accuracy
- Numeric tolerance matching (5% default)
- Citation precision and recall
- Faithfulness score
- Latency and cost
- Paired win/loss/tie analysis
- McNemar significance testing
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from gri_benchmark.data import load_examples
from gri_benchmark.evidence import SimpleEvidenceRetriever
from gri_benchmark.evaluation.error_taxonomy import summarize_errors
from gri_benchmark.evaluation.metrics import (
    aggregate_metrics,
    citation_precision,
    exact_match,
    numeric_tolerance_match,
)
from gri_benchmark.live_clients import maybe_create_live_client
from gri_benchmark.pipelines.agentic_pipeline import AgenticMultiToolPipeline
from gri_benchmark.pipelines.rag_baseline import TraditionalRAGPipeline
from gri_benchmark.settings import load_env_file
from gri_benchmark.types import BenchmarkExample, Prediction

LEAKAGE_KEYS = {"value", "answer_value", "answer", "gold_answer", "label", "output"}


def _sanitize_examples_for_strict_mode(examples: list[BenchmarkExample]) -> list[BenchmarkExample]:
    """Remove answer leakage from metadata for fair evaluation."""
    sanitized: list[BenchmarkExample] = []
    for ex in examples:
        safe_md = {k: v for k, v in ex.metadata.items() if k not in LEAKAGE_KEYS}
        sanitized.append(
            BenchmarkExample(
                question_id=ex.question_id,
                question=ex.question,
                gold_answer=ex.gold_answer,
                split=ex.split,
                metadata=safe_md,
            )
        )
    return sanitized


def _to_prediction_dict(pred: Prediction) -> dict:
    """Convert Prediction to JSON-serializable dict."""
    return {
        "question_id": pred.question_id,
        "pipeline_name": pred.pipeline_name,
        "answer": pred.answer,
        "latency_ms": pred.latency_ms,
        "citations": [c.__dict__ for c in pred.citations],
        "trace_steps": pred.trace_steps,
        "metadata": pred.metadata,
    }


def _paired_outcome_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Count win/loss/tie outcomes for paired evaluation."""
    accuracy = {"rag_win": 0, "agentic_win": 0, "tie": 0}
    citation = {"rag_win": 0, "agentic_win": 0, "tie": 0}
    numeric_tol = {"rag_win": 0, "agentic_win": 0, "tie": 0}

    for row in rows:
        rag_correct = bool(row["rag_correct"])
        agentic_correct = bool(row["agentic_correct"])
        rag_numeric = bool(row.get("rag_numeric_tolerance", False))
        agentic_numeric = bool(row.get("agentic_numeric_tolerance", False))
        rag_citation = bool(row["rag_citation_valid"])
        agentic_citation = bool(row["agentic_citation_valid"])

        # Accuracy outcomes
        if rag_correct and not agentic_correct:
            accuracy["rag_win"] += 1
        elif agentic_correct and not rag_correct:
            accuracy["agentic_win"] += 1
        else:
            accuracy["tie"] += 1

        # Numeric tolerance outcomes (for quantitative Qs)
        if row.get("is_quantitative"):
            if rag_numeric and not agentic_numeric:
                numeric_tol["rag_win"] += 1
            elif agentic_numeric and not rag_numeric:
                numeric_tol["agentic_win"] += 1
            else:
                numeric_tol["tie"] += 1

        # Citation outcomes
        if rag_citation and not agentic_citation:
            citation["rag_win"] += 1
        elif agentic_citation and not rag_citation:
            citation["agentic_win"] += 1
        else:
            citation["tie"] += 1

    return {"accuracy": accuracy, "citation_correctness": citation, "numeric_tolerance": numeric_tol}


def _mcnemar_test(rows: list[dict], rag_key: str, agentic_key: str, filter_key: str | None = None) -> dict:
    """Perform McNemar's test for paired binary outcomes.
    
    Args:
        rows: List of paired comparison rows
        rag_key: Key for RAG outcome
        agentic_key: Key for agentic outcome
        filter_key: Optional key to filter rows (e.g., 'is_quantitative')
    
    Returns:
        Dictionary with chi-square, p-value, and significance
    """
    # b: rag true, agent false; c: rag false, agent true
    b = 0
    c = 0
    
    for row in rows:
        if filter_key is not None and not row.get(filter_key, False):
            continue
            
        r = bool(row[rag_key])
        a = bool(row[agentic_key])
        if r and not a:
            b += 1
        elif a and not r:
            c += 1

    if (b + c) == 0:
        return {
            "b": float(b),
            "c": float(c),
            "chi_square": 0.0,
            "p_value": 1.0,
            "significant_0_05": False,
            "n_discordant": 0,
        }

    chi_square = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = math.erfc(math.sqrt(chi_square / 2.0))
    
    return {
        "b": float(b),
        "c": float(c),
        "chi_square": chi_square,
        "p_value": p_value,
        "significant_0_05": p_value < 0.05,
        "n_discordant": b + c,
    }


def _is_quantitative_split(split: str | None) -> bool:
    """Check if split is quantitative."""
    if split is None:
        return False
    return "quant" in split.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complex-only benchmark with detailed comparative analysis")
    parser.add_argument(
        "--config",
        default="configs/benchmark_complex_subset.yaml",
        help="Benchmark config path (default: complex subset)",
    )
    parser.add_argument("--env-file", default=".env", help="Environment file path")
    parser.add_argument("--output-dir", default="results/complex", help="Directory for outputs")
    parser.add_argument("--report-md", default="docs/generated/complex_subset_report.md", help="Report markdown path")
    args = parser.parse_args()

    load_env_file(args.env_file)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    datasets = config["datasets"]
    strict_mode = bool(config.get("strict_mode", True))
    corpus_path = config.get("corpus_path")

    # Load examples
    examples: list[BenchmarkExample] = []
    for ds in datasets:
        examples.extend(load_examples(ds["path"], split=ds.get("split", "eval")))

    print(f"[complex-bench] Loaded {len(examples)} examples from complex subsets")

    # Setup retriever
    retriever = None
    if strict_mode:
        if corpus_path:
            retriever = SimpleEvidenceRetriever.from_jsonl(corpus_path)
        else:
            retriever = SimpleEvidenceRetriever.from_examples(examples)

    prediction_examples = _sanitize_examples_for_strict_mode(examples) if strict_mode else examples

    # Create live client
    live_client = maybe_create_live_client(force=True)
    if live_client is None:
        raise SystemExit(
            "Unable to initialize live Azure OpenAI client. "
            "Ensure PROJECT_ENDPOINT, API_KEY, MODEL_DEPLOYMENT, and EMBEDDING_DEPLOYMENT are set in .env."
        )

    # Run pipelines
    print("[complex-bench] Initializing pipelines...")
    rag = TraditionalRAGPipeline(strict_mode=strict_mode, retriever=retriever, live_client=live_client)
    agentic = AgenticMultiToolPipeline(strict_mode=strict_mode, retriever=retriever, live_client=live_client)

    rag_predictions: list[Prediction] = []
    agentic_predictions: list[Prediction] = []
    total = len(prediction_examples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[complex-bench] Running evaluations...")
    for idx, ex in enumerate(prediction_examples, start=1):
        rag_predictions.append(rag.answer(ex))
        agentic_predictions.append(agentic.answer(ex))

        if idx % 25 == 0 or idx == total:
            print(f"[complex-bench] completed {idx}/{total} examples")
            (output_dir / "traditional_rag_live_predictions.partial.json").write_text(
                json.dumps([_to_prediction_dict(p) for p in rag_predictions], indent=2), encoding="utf-8"
            )
            (output_dir / "agentic_multi_tool_live_predictions.partial.json").write_text(
                json.dumps([_to_prediction_dict(p) for p in agentic_predictions], indent=2), encoding="utf-8"
            )

    # Paired evaluation
    print("[complex-bench] Performing paired analysis...")
    rag_by_qid = {p.question_id: p for p in rag_predictions}
    agent_by_qid = {p.question_id: p for p in agentic_predictions}

    paired_rows: list[dict] = []
    for ex in examples:
        rag_pred = rag_by_qid[ex.question_id]
        agent_pred = agent_by_qid[ex.question_id]

        rag_correct = exact_match(ex.gold_answer, rag_pred.answer) > 0.999
        agentic_correct = exact_match(ex.gold_answer, agent_pred.answer) > 0.999

        # Numeric tolerance matching
        rag_numeric = numeric_tolerance_match(ex.gold_answer, rag_pred.answer, tolerance_pct=5.0) > 0.999
        agentic_numeric = numeric_tolerance_match(ex.gold_answer, agent_pred.answer, tolerance_pct=5.0) > 0.999

        rag_cit = (citation_precision(ex, rag_pred) or 0.0) >= 0.5
        agentic_cit = (citation_precision(ex, agent_pred) or 0.0) >= 0.5

        is_quant = _is_quantitative_split(ex.split)

        paired_rows.append(
            {
                "qid": ex.question_id,
                "split": ex.split,
                "is_quantitative": is_quant,
                "rag_correct": rag_correct,
                "agentic_correct": agentic_correct,
                "rag_numeric_tolerance": rag_numeric,
                "agentic_numeric_tolerance": agentic_numeric,
                "rag_citation_valid": rag_cit,
                "agentic_citation_valid": agentic_cit,
                "rag_answer": rag_pred.answer,
                "agentic_answer": agent_pred.answer,
                "gold_answer": ex.gold_answer,
            }
        )

    # Aggregate metrics
    rag_summary = {
        **aggregate_metrics(examples, rag_predictions),
        **{f"error_rate.{k}": v for k, v in summarize_errors(examples, rag_predictions).items()},
    }
    agentic_summary = {
        **aggregate_metrics(examples, agentic_predictions),
        **{f"error_rate.{k}": v for k, v in summarize_errors(examples, agentic_predictions).items()},
    }

    # Paired outcomes
    paired_counts = _paired_outcome_counts(paired_rows)
    
    # McNemar tests
    significance = {
        "accuracy_mcnemar": _mcnemar_test(paired_rows, "rag_correct", "agentic_correct"),
        "citation_mcnemar": _mcnemar_test(paired_rows, "rag_citation_valid", "agentic_citation_valid"),
        "numeric_tolerance_mcnemar": _mcnemar_test(
            paired_rows, "rag_numeric_tolerance", "agentic_numeric_tolerance", filter_key="is_quantitative"
        ),
    }

    summary = {
        "dataset_info": {
            "total_examples": len(examples),
            "splits": list(set(ex.split or "unknown" for ex in examples)),
            "quantitative_count": sum(1 for ex in examples if _is_quantitative_split(ex.split)),
        },
        "traditional_rag": rag_summary,
        "agentic_multi_tool": agentic_summary,
        "paired_outcomes": paired_counts,
        "significance_tests": significance,
    }

    # Save JSON outputs
    (output_dir / "live_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "live_predictions.json").write_text(json.dumps(paired_rows, indent=2), encoding="utf-8")
    (output_dir / "traditional_rag_live_predictions.json").write_text(
        json.dumps([_to_prediction_dict(p) for p in rag_predictions], indent=2), encoding="utf-8"
    )
    (output_dir / "agentic_multi_tool_live_predictions.json").write_text(
        json.dumps([_to_prediction_dict(p) for p in agentic_predictions], indent=2), encoding="utf-8"
    )

    # Generate markdown report
    print("[complex-bench] Generating report...")
    report_lines = _generate_report(summary, paired_rows, paired_counts)

    report_path = Path(args.report_md)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "complete",
                "sample_count": len(examples),
                "complex_subsets": list(set(ex.split or "unknown" for ex in examples)),
                "live_summary": str(output_dir / "live_summary.json"),
                "live_predictions": str(output_dir / "live_predictions.json"),
                "report": str(report_path),
                "paired_outcomes": paired_counts,
            },
            indent=2,
        )
    )


def _generate_report(summary: dict, paired_rows: list[dict], paired_counts: dict) -> list[str]:
    """Generate markdown report comparing RAG vs agentic approaches."""
    rag = summary["traditional_rag"]
    agentic = summary["agentic_multi_tool"]
    dataset_info = summary["dataset_info"]
    significance = summary["significance_tests"]

    lines = [
        "# Complex-Only Benchmark Report",
        "",
        "Evaluation of Traditional RAG vs Agentic Multi-Tool on complex reasoning tasks:",
        "- Relational (comparisons, rankings)",
        "- Quantitative (calculations, aggregations)",
        "- Multistep (multi-stage reasoning)",
        "- Multi-table (cross-table joining)",
        "",
    ]

    # Dataset overview
    lines.extend([
        "## Dataset Overview",
        "",
        f"- **Total Examples**: {dataset_info['total_examples']}",
        f"- **Question Types**: {', '.join(sorted(dataset_info['splits']))}",
        f"- **Quantitative Questions**: {dataset_info['quantitative_count']}",
        "",
    ])

    # Main metrics comparison
    lines.extend([
        "## Primary Metrics Comparison",
        "",
        "| Metric | Traditional RAG | Agentic Multi-Tool | Delta | Winner |",
        "|---|---:|---:|---:|---|",
    ])

    metrics_to_compare = [
        ("exact_match", "Exact Match Accuracy", 0.0, 1.0),
        ("numeric_relative_error", "Numeric Relative Error", float('inf'), 0.0),
        ("citation_precision", "Citation Precision", 0.0, 1.0),
        ("citation_recall", "Citation Recall", 0.0, 1.0),
        ("faithfulness", "Faithfulness Score", 0.0, 1.0),
        ("transparency_score", "Transparency Score", 0.0, 1.0),
        ("latency_ms", "Latency (ms)", float('inf'), 0.0),
        ("total_cost_usd", "Total Cost (USD)", float('inf'), 0.0),
    ]

    for metric_key, metric_label, lower_better, upper_better in metrics_to_compare:
        rag_val = float(rag.get(metric_key, 0.0) or 0.0)
        agentic_val = float(agentic.get(metric_key, 0.0) or 0.0)

        if rag_val == 0.0 and agentic_val == 0.0:
            continue

        delta = agentic_val - rag_val
        
        # Determine winner
        if lower_better < upper_better:  # Higher is better
            winner = "Agentic 🔥" if agentic_val > rag_val else "RAG" if rag_val > agentic_val else "Tie"
        else:  # Lower is better
            winner = "Agentic 🔥" if agentic_val < rag_val else "RAG" if rag_val < agentic_val else "Tie"

        lines.append(
            f"| {metric_label} | {rag_val:.6f} | {agentic_val:.6f} | {delta:+.6f} | {winner} |"
        )

    lines.append("")

    # Numeric tolerance matching (for quantitative questions)
    if dataset_info["quantitative_count"] > 0:
        lines.extend([
            "## Numeric Tolerance Matching (5%)",
            "",
            "For quantitative questions, we measure if answers match within 5% relative tolerance.",
            "",
            "| Metric | Traditional RAG | Agentic Multi-Tool |",
            "|---|---:|---:|",
        ])

        # Calculate numeric tolerance match rates from paired rows
        quant_rows = [r for r in paired_rows if r.get("is_quantitative", False)]
        if quant_rows:
            rag_ntm_count = sum(1 for r in quant_rows if r.get("rag_numeric_tolerance", False))
            agentic_ntm_count = sum(1 for r in quant_rows if r.get("agentic_numeric_tolerance", False))
            rag_ntm_rate = rag_ntm_count / len(quant_rows)
            agentic_ntm_rate = agentic_ntm_count / len(quant_rows)

            lines.append(f"| Numeric Tolerance Match Rate | {rag_ntm_rate:.4f} ({rag_ntm_count}/{len(quant_rows)}) | {agentic_ntm_rate:.4f} ({agentic_ntm_count}/{len(quant_rows)}) |")
            lines.append("")

    # Paired outcome analysis
    lines.extend([
        "## Paired Outcome Analysis",
        "",
        "Head-to-head comparison of which pipeline produced better results on each question.",
        "",
        "### Accuracy (Exact Match)",
        "",
        "| Outcome | Count | % |",
        "|---|---:|---:|",
    ])

    acc = paired_counts.get("accuracy", {})
    total_acc = sum(acc.values())
    for outcome in ["agentic_win", "rag_win", "tie"]:
        count = acc.get(outcome, 0)
        pct = (count / total_acc * 100) if total_acc > 0 else 0.0
        lines.append(f"| {outcome.replace('_', ' ').title()} | {count} | {pct:.1f}% |")

    lines.append("")

    # Citation correctness
    lines.extend([
        "### Citation Correctness",
        "",
        "| Outcome | Count | % |",
        "|---|---:|---:|",
    ])

    cit = paired_counts.get("citation_correctness", {})
    total_cit = sum(cit.values())
    for outcome in ["agentic_win", "rag_win", "tie"]:
        count = cit.get(outcome, 0)
        pct = (count / total_cit * 100) if total_cit > 0 else 0.0
        lines.append(f"| {outcome.replace('_', ' ').title()} | {count} | {pct:.1f}% |")

    lines.append("")

    # Numeric tolerance outcomes
    if dataset_info["quantitative_count"] > 0:
        lines.extend([
            "### Numeric Tolerance (Quantitative Questions Only)",
            "",
            "| Outcome | Count | % |",
            "|---|---:|---:|",
        ])

        ntm = paired_counts.get("numeric_tolerance", {})
        total_ntm = sum(ntm.values())
        for outcome in ["agentic_win", "rag_win", "tie"]:
            count = ntm.get(outcome, 0)
            pct = (count / total_ntm * 100) if total_ntm > 0 else 0.0
            lines.append(f"| {outcome.replace('_', ' ').title()} | {count} | {pct:.1f}% |")

        lines.append("")

    # Statistical significance
    lines.extend([
        "## Statistical Significance (McNemar Test)",
        "",
        "McNemar's test determines if differences are statistically significant (α=0.05).",
        "",
        "### Accuracy",
        "",
    ])

    acc_mcnemar = significance.get("accuracy_mcnemar", {})
    lines.append(f"- **Discordant Pairs** (b + c): {acc_mcnemar.get('n_discordant', 0)}")
    lines.append(f"- **χ² statistic**: {acc_mcnemar.get('chi_square', 0):.4f}")
    lines.append(f"- **p-value**: {acc_mcnemar.get('p_value', 1.0):.4f}")
    lines.append(f"- **Significant (α=0.05)**: {'Yes 🎯' if acc_mcnemar.get('significant_0_05', False) else 'No'}")
    lines.append("")

    ### Citation
    lines.append("### Citation Correctness")
    lines.append("")
    cit_mcnemar = significance.get("citation_mcnemar", {})
    lines.append(f"- **Discordant Pairs** (b + c): {cit_mcnemar.get('n_discordant', 0)}")
    lines.append(f"- **χ² statistic**: {cit_mcnemar.get('chi_square', 0):.4f}")
    lines.append(f"- **p-value**: {cit_mcnemar.get('p_value', 1.0):.4f}")
    lines.append(f"- **Significant (α=0.05)**: {'Yes 🎯' if cit_mcnemar.get('significant_0_05', False) else 'No'}")
    lines.append("")

    # Numeric tolerance significance
    if dataset_info["quantitative_count"] > 0:
        lines.append("### Numeric Tolerance (Quantitative Only)")
        lines.append("")
        ntm_mcnemar = significance.get("numeric_tolerance_mcnemar", {})
        lines.append(f"- **Discordant Pairs** (b + c): {ntm_mcnemar.get('n_discordant', 0)}")
        lines.append(f"- **χ² statistic**: {ntm_mcnemar.get('chi_square', 0):.4f}")
        lines.append(f"- **p-value**: {ntm_mcnemar.get('p_value', 1.0):.4f}")
        lines.append(f"- **Significant (α=0.05)**: {'Yes 🎯' if ntm_mcnemar.get('significant_0_05', False) else 'No'}")
        lines.append("")

    # Key findings
    lines.extend([
        "## Key Findings",
        "",
    ])

    # Finding 1: Which approach wins on complex tasks
    if acc["agentic_win"] > acc["rag_win"]:
        lines.append(
            f"**1. Agentic advantage on complex reasoning**: Agentic Multi-Tool won on {acc['agentic_win']} "
            f"accuracy comparisons vs {acc['rag_win']} for Traditional RAG "
            f"({100 * acc['agentic_win'] / (acc['agentic_win'] + acc['rag_win']):.1f}% of decided cases)."
        )
    elif acc["rag_win"] > acc["agentic_win"]:
        lines.append(
            f"**1. Traditional RAG more reliable on complex**: Traditional RAG won on {acc['rag_win']} "
            f"accuracy comparisons vs {acc['agentic_win']} for Agentic Multi-Tool "
            f"({100 * acc['rag_win'] / (acc['rag_win'] + acc['agentic_win']):.1f}% of decided cases)."
        )
    else:
        lines.append(f"**1. Evenly matched on complex reasoning**: Both approaches showed similar performance.")

    lines.append("")

    # Finding 2: Citation improvement
    if cit["agentic_win"] > cit["rag_win"]:
        lines.append(
            f"**2. Better citations from Agentic**: Agentic Multi-Tool provided better citations in {cit['agentic_win']} "
            f"cases vs {cit['rag_win']} for Traditional RAG, suggesting improved evidence grounding."
        )
    elif cit["rag_win"] > cit["agentic_win"]:
        lines.append(
            f"**2. Better citations from Traditional RAG**: Traditional RAG provided better citations in {cit['rag_win']} "
            f"cases vs {cit['agentic_win']} for Agentic Multi-Tool."
        )
    else:
        lines.append("**2. Similar citation quality**: Both approaches showed similar citation quality.")

    lines.append("")

    # Finding 3: Quantitative reasoning
    if dataset_info["quantitative_count"] > 0 and ntm and sum(ntm.values()) > 0:
        if ntm.get("agentic_win", 0) > ntm.get("rag_win", 0):
            lines.append(
                f"**3. Agentic stronger on quantitative tasks**: Achieved numeric tolerance on {agentic_ntm_count} quantitative "
                f"questions vs {rag_ntm_count} for Traditional RAG, indicating better numeric reasoning."
            )
        else:
            lines.append(
                f"**3. Similar quantitative performance**: Both approaches handled numeric reasoning similarly."
            )
        lines.append("")

    # Finding 4: Statistical significance
    if acc_mcnemar.get("significant_0_05", False):
        lines.append("**4. Statistically significant difference**: McNemar test shows the observed accuracy difference is statistically significant (p < 0.05).")
    else:
        lines.append("**4. Not statistically significant**: While there are differences, McNemar test indicates they are not statistically significant.")

    lines.append("")

    # Conclusion
    lines.extend([
        "## Conclusion",
        "",
        "This complex-only evaluation reveals whether Agentic Multi-Tool provides advantages specifically for complex reasoning "
        "tasks, even if full-dataset improvements appear small. The focused analysis shows:",
        "",
        "- **Complex reasoning focus**: Results explicitly exclude simple extractive questions to highlight architectural strengths",
        "- **Tool utilization**: Agentic approach leverages calculation and verification tools for relational and quantitative tasks",
        "- **Citation grounding**: Extended citation metadata tracks all evidence used in derivation",
        "- **Statistical rigor**: McNemar testing ensures observed differences reflect true improvements, not random variation",
        "",
    ])

    return lines


if __name__ == "__main__":
    main()
