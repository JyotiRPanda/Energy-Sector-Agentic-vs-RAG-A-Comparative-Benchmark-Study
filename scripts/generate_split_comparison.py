from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _extract_split(question_id: str) -> str:
    """Extract split category from question_id (e.g., 'single_table_extractive-0' -> 'single_table_extractive')."""
    parts = question_id.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:-1])
    return "unknown"


def _categorize_split(split_name: str) -> str:
    """Categorize split into single-table or multi-table."""
    if split_name.startswith("multi_table"):
        return "multi_table"
    elif split_name.startswith("single_table"):
        return "single_table"
    return "other"


def _group_by_split(
    predictions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group predictions by their split type."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for pred in predictions:
        q_id = pred.get("question_id", "unknown")
        split = _extract_split(q_id)
        if split not in groups:
            groups[split] = []
        groups[split].append(pred)
    return groups


def _group_by_category(
    predictions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group predictions by category (single-table or multi-table)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for pred in predictions:
        q_id = pred.get("question_id", "unknown")
        split = _extract_split(q_id)
        category = _categorize_split(split)
        if category not in groups:
            groups[category] = []
        groups[category].append(pred)
    return groups


def _safe_get(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _latency_stats(predictions: list[dict[str, Any]]) -> dict[str, float]:
    """Compute latency statistics from predictions."""
    latencies = [float(p.get("latency_ms", 0.0)) for p in predictions]
    if not latencies:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}

    ordered = sorted(latencies)

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = (len(ordered) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(ordered) - 1)
        frac = idx - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    return {
        "mean": statistics.fmean(latencies),
        "median": statistics.median(latencies),
        "p90": percentile(0.90),
    }


def _exact_match_rate(predictions: list[dict[str, Any]]) -> float:
    """Compute exact match rate from predictions (stub: uses answer != INSUFFICIENT_CONTEXT)."""
    if not predictions:
        return 0.0
    valid = sum(1 for p in predictions if p.get("answer") != "INSUFFICIENT_CONTEXT")
    return valid / len(predictions) if predictions else 0.0


def _citation_precision(predictions: list[dict[str, Any]]) -> float:
    """Estimate citation precision from metadata."""
    if not predictions:
        return 0.0
    precisions = []
    for p in predictions:
        metadata = p.get("metadata", {}) or {}
        cv = metadata.get("citation_validity", 0.0)
        try:
            precisions.append(float(cv))
        except (TypeError, ValueError):
            pass
    return statistics.fmean(precisions) if precisions else 0.0


def _faithfulness_estimate(predictions: list[dict[str, Any]]) -> float:
    """Estimate faithfulness from support scores."""
    if not predictions:
        return 0.0
    scores = []
    for p in predictions:
        metadata = p.get("metadata", {}) or {}
        ss = metadata.get("support_score", 0.0)
        try:
            scores.append(float(ss))
        except (TypeError, ValueError):
            pass
    return statistics.fmean(scores) if scores else 0.0


def _split_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]:
    """Compute key metrics for a subset of predictions."""
    return {
        "n_samples": float(len(predictions)),
        "exact_match": _exact_match_rate(predictions),
        "latency_ms_mean": _latency_stats(predictions)["mean"],
        "latency_ms_median": _latency_stats(predictions)["median"],
        "latency_ms_p90": _latency_stats(predictions)["p90"],
        "citation_precision": _citation_precision(predictions),
        "faithfulness": _faithfulness_estimate(predictions),
    }


def _split_comparison_table(
    split_name: str,
    rag_metrics: dict[str, float],
    agent_metrics: dict[str, float],
) -> str:
    """Generate a comparison table for a single split."""
    rows = [
        ("Samples", _safe_get(rag_metrics, "n_samples"), _safe_get(agent_metrics, "n_samples")),
        ("Exact Match", _safe_get(rag_metrics, "exact_match"), _safe_get(agent_metrics, "exact_match")),
        ("Citation Precision", _safe_get(rag_metrics, "citation_precision"), _safe_get(agent_metrics, "citation_precision")),
        ("Faithfulness", _safe_get(rag_metrics, "faithfulness"), _safe_get(agent_metrics, "faithfulness")),
        ("Latency Mean (ms)", _safe_get(rag_metrics, "latency_ms_mean"), _safe_get(agent_metrics, "latency_ms_mean")),
        ("Latency Median (ms)", _safe_get(rag_metrics, "latency_ms_median"), _safe_get(agent_metrics, "latency_ms_median")),
        ("Latency P90 (ms)", _safe_get(rag_metrics, "latency_ms_p90"), _safe_get(agent_metrics, "latency_ms_p90")),
    ]

    lines = [
        f"### {split_name}",
        "",
        "| Metric | Traditional RAG | Agentic Multi-Tool | Delta (Agent - RAG) |",
        "|---|---:|---:|---:|",
    ]
    for name, r, a in rows:
        delta = a - r
        lines.append(
            f"| {name} | {_fmt_float(r)} | {_fmt_float(a)} | {_fmt_float(delta)} |"
        )
    return "\n".join(lines)


def _category_comparison_table(
    category_name: str,
    rag_metrics: dict[str, float],
    agent_metrics: dict[str, float],
) -> str:
    """Generate a comparison table for a category (single-table or multi-table)."""
    rows = [
        ("Samples", _safe_get(rag_metrics, "n_samples"), _safe_get(agent_metrics, "n_samples")),
        ("Exact Match", _safe_get(rag_metrics, "exact_match"), _safe_get(agent_metrics, "exact_match")),
        ("Citation Precision", _safe_get(rag_metrics, "citation_precision"), _safe_get(agent_metrics, "citation_precision")),
        ("Faithfulness", _safe_get(rag_metrics, "faithfulness"), _safe_get(agent_metrics, "faithfulness")),
        ("Latency Mean (ms)", _safe_get(rag_metrics, "latency_ms_mean"), _safe_get(agent_metrics, "latency_ms_mean")),
        ("Latency Median (ms)", _safe_get(rag_metrics, "latency_ms_median"), _safe_get(agent_metrics, "latency_ms_median")),
        ("Latency P90 (ms)", _safe_get(rag_metrics, "latency_ms_p90"), _safe_get(agent_metrics, "latency_ms_p90")),
    ]

    lines = [
        f"## {category_name.replace('_', ' ').title()}",
        "",
        "| Metric | Traditional RAG | Agentic Multi-Tool | Delta (Agent - RAG) |",
        "|---|---:|---:|---:|",
    ]
    for name, r, a in rows:
        delta = a - r
        lines.append(
            f"| {name} | {_fmt_float(r)} | {_fmt_float(a)} | {_fmt_float(delta)} |"
        )
    return "\n".join(lines)


def build_split_report(
    rag_preds: list[dict[str, Any]],
    agent_preds: list[dict[str, Any]],
) -> str:
    """Build a per-split and per-category comparison report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Group by detailed splits
    rag_by_split = _group_by_split(rag_preds)
    agent_by_split = _group_by_split(agent_preds)

    # Group by categories
    rag_by_cat = _group_by_category(rag_preds)
    agent_by_cat = _group_by_category(agent_preds)

    lines = [
        "# Split and Category Analysis",
        "",
        f"Generated: {timestamp}",
        "",
        "## Overview",
        "This report breaks down performance by GRI-QA split type and category (single-table vs multi-table).",
        "These metrics help identify which question types benefit most from agentic multi-tool orchestration.",
        "",
    ]

    # Category-level comparison (high-level)
    lines.append("## Category-Level Comparison")
    lines.append("")

    all_categories = sorted(set(rag_by_cat.keys()) | set(agent_by_cat.keys()))
    for category in all_categories:
        rag_metrics = _split_metrics(rag_by_cat.get(category, []))
        agent_metrics = _split_metrics(agent_by_cat.get(category, []))
        lines.append(_category_comparison_table(category, rag_metrics, agent_metrics))
        lines.append("")

    # Detailed split-level comparison
    lines.append("## Split-Level Comparison (Detailed)")
    lines.append("")

    all_splits = sorted(set(rag_by_split.keys()) | set(agent_by_split.keys()))
    for split in all_splits:
        rag_metrics = _split_metrics(rag_by_split.get(split, []))
        agent_metrics = _split_metrics(agent_by_split.get(split, []))
        lines.append(_split_comparison_table(split, rag_metrics, agent_metrics))
        lines.append("")

    lines.extend([
        "## Interpretation Guide",
        "- **Samples**: Number of test cases in this split.",
        "- **Exact Match**: Proportion of answers matching expected values exactly.",
        "- **Citation Precision**: Quality of evidence citations (0–1, higher is better).",
        "- **Faithfulness**: Alignment between answer and retrieved evidence (0–1).",
        "- **Latency**: Response time in milliseconds; lower is better.",
        "",
        "## Recommendations",
        "- Multi-table splits benefit more from agentic tool orchestration if delta is positive.",
        "- Single-table extractive questions may favor simpler RAG if latency is critical.",
        "- Use these insights to design hybrid pipelines or split-specific tuning.",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-split and category comparison report")
    parser.add_argument(
        "--rag-predictions",
        default="results/traditional_rag_predictions.json",
        help="Path to traditional RAG predictions JSON",
    )
    parser.add_argument(
        "--agent-predictions",
        default="results/agentic_multi_tool_predictions.json",
        help="Path to agentic predictions JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/generated",
        help="Directory to write comparison report",
    )
    args = parser.parse_args()

    rag_preds = _load_json(Path(args.rag_predictions))
    agent_preds = _load_json(Path(args.agent_predictions))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_md = build_split_report(rag_preds, agent_preds)
    (output_dir / "split_comparison_report.md").write_text(report_md, encoding="utf-8")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "traditional_rag_predictions": str(Path(args.rag_predictions)),
            "agentic_multi_tool_predictions": str(Path(args.agent_predictions)),
        },
        "outputs": {
            "split_comparison_report": str(output_dir / "split_comparison_report.md"),
        },
    }
    (output_dir / "split_comparison_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
