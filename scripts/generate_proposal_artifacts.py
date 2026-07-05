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


def _safe_get(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _latency_stats(predictions: list[dict[str, Any]]) -> dict[str, float]:
    latencies = [float(p.get("latency_ms", 0.0)) for p in predictions]
    if not latencies:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

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
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _tool_step_stats(predictions: list[dict[str, Any]]) -> dict[str, float]:
    steps = []
    for p in predictions:
        metadata = p.get("metadata", {}) or {}
        if "tool_steps" in metadata:
            try:
                steps.append(float(metadata["tool_steps"]))
            except (TypeError, ValueError):
                pass

    if not steps:
        return {"mean": 0.0, "max": 0.0}

    return {
        "mean": statistics.fmean(steps),
        "max": max(steps),
    }


def _error_rates(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith("error_rate."):
            label = key.split(".", 1)[1]
            try:
                out[label] = float(value)
            except (TypeError, ValueError):
                out[label] = 0.0
    return dict(sorted(out.items(), key=lambda x: x[0]))


def _comparison_table(summary: dict[str, dict[str, Any]]) -> str:
    rag = summary["traditional_rag"]
    agent = summary["agentic_multi_tool"]

    rows = [
        ("Exact Match", _safe_get(rag, "exact_match"), _safe_get(agent, "exact_match")),
        ("Numeric Relative Error", _safe_get(rag, "numeric_relative_error"), _safe_get(agent, "numeric_relative_error")),
        ("Citation Precision", _safe_get(rag, "citation_precision"), _safe_get(agent, "citation_precision")),
        ("Citation Recall", _safe_get(rag, "citation_recall"), _safe_get(agent, "citation_recall")),
        ("Faithfulness", _safe_get(rag, "faithfulness"), _safe_get(agent, "faithfulness")),
        ("Transparency", _safe_get(rag, "transparency"), _safe_get(agent, "transparency")),
        ("Latency (ms)", _safe_get(rag, "latency_ms"), _safe_get(agent, "latency_ms")),
    ]

    lines = [
        "| Metric | Traditional RAG | Agentic Multi-Tool | Delta (Agent - RAG) |",
        "|---|---:|---:|---:|",
    ]
    for name, r, a in rows:
        delta = a - r
        lines.append(
            f"| {name} | {_fmt_float(r)} | {_fmt_float(a)} | {_fmt_float(delta)} |"
        )
    return "\n".join(lines)


def _error_table(summary: dict[str, dict[str, Any]]) -> str:
    rag_errors = _error_rates(summary["traditional_rag"])
    agent_errors = _error_rates(summary["agentic_multi_tool"])
    labels = sorted(set(rag_errors) | set(agent_errors))

    lines = [
        "| Error Type | Traditional RAG | Agentic Multi-Tool | Delta (Agent - RAG) |",
        "|---|---:|---:|---:|",
    ]

    for label in labels:
        r = rag_errors.get(label, 0.0)
        a = agent_errors.get(label, 0.0)
        lines.append(
            f"| {label} | {_fmt_float(r)} | {_fmt_float(a)} | {_fmt_float(a - r)} |"
        )
    return "\n".join(lines)


def _latency_table(rag_preds: list[dict[str, Any]], agent_preds: list[dict[str, Any]]) -> str:
    rag = _latency_stats(rag_preds)
    agent = _latency_stats(agent_preds)

    lines = [
        "| Latency Statistic (ms) | Traditional RAG | Agentic Multi-Tool |",
        "|---|---:|---:|",
        f"| Mean | {_fmt_float(rag['mean'])} | {_fmt_float(agent['mean'])} |",
        f"| Median | {_fmt_float(rag['median'])} | {_fmt_float(agent['median'])} |",
        f"| P90 | {_fmt_float(rag['p90'])} | {_fmt_float(agent['p90'])} |",
        f"| P95 | {_fmt_float(rag['p95'])} | {_fmt_float(agent['p95'])} |",
        f"| Max | {_fmt_float(rag['max'])} | {_fmt_float(agent['max'])} |",
    ]
    return "\n".join(lines)


def _trace_table(rag_preds: list[dict[str, Any]], agent_preds: list[dict[str, Any]]) -> str:
    rag_tools = _tool_step_stats(rag_preds)
    agent_tools = _tool_step_stats(agent_preds)

    lines = [
        "| Trace/Tool Statistic | Traditional RAG | Agentic Multi-Tool |",
        "|---|---:|---:|",
        f"| Mean Tool Steps | {_fmt_float(rag_tools['mean'])} | {_fmt_float(agent_tools['mean'])} |",
        f"| Max Tool Steps | {_fmt_float(rag_tools['max'])} | {_fmt_float(agent_tools['max'])} |",
    ]
    return "\n".join(lines)


def _mermaid_bar(label: str, rag_value: float, agent_value: float) -> str:
    return f'    "{label}" : {rag_value:.6f}, {agent_value:.6f}'


def _build_mermaid(summary: dict[str, dict[str, Any]]) -> str:
    rag = summary["traditional_rag"]
    agent = summary["agentic_multi_tool"]

    lines = [
        "xychart-beta",
        "    title \"RAG vs Agentic Core Metrics\"",
        "    x-axis [ExactMatch, CitationPrecision, Faithfulness, Transparency]",
        "    y-axis \"Score\" 0 --> 1",
        "    bar ["
        f"{_safe_get(rag, 'exact_match'):.6f}, "
        f"{_safe_get(rag, 'citation_precision'):.6f}, "
        f"{_safe_get(rag, 'faithfulness'):.6f}, "
        f"{_safe_get(rag, 'transparency'):.6f}"
        "]",
        "    bar ["
        f"{_safe_get(agent, 'exact_match'):.6f}, "
        f"{_safe_get(agent, 'citation_precision'):.6f}, "
        f"{_safe_get(agent, 'faithfulness'):.6f}, "
        f"{_safe_get(agent, 'transparency'):.6f}"
        "]",
        "",
        "xychart-beta",
        "    title \"Latency (ms): Lower is Better\"",
        "    x-axis [TraditionalRAG, AgenticMultiTool]",
        "    y-axis \"Latency ms\" 0 --> 0.010",
        f"    bar [{_safe_get(rag, 'latency_ms'):.6f}, {_safe_get(agent, 'latency_ms'):.6f}]",
    ]
    return "\n".join(lines)


def build_report(
    summary: dict[str, dict[str, Any]],
    rag_preds: list[dict[str, Any]],
    agent_preds: list[dict[str, Any]],
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    rag_n = int(_safe_get(summary["traditional_rag"], "n_samples"))
    agent_n = int(_safe_get(summary["agentic_multi_tool"], "n_samples"))

    return "\n".join(
        [
            "# Benchmark Artifact Pack",
            "",
            f"Generated: {timestamp}",
            "",
            "## Scope",
            "This report compares traditional RAG and agentic multi-tool pipelines for table-grounded QA.",
            "No public API deployment is required to reproduce these results.",
            "",
            "## Sample Counts",
            f"- Traditional RAG: {rag_n}",
            f"- Agentic Multi-Tool: {agent_n}",
            "",
            "## Aggregate Metrics",
            _comparison_table(summary),
            "",
            "## Error Pattern Comparison",
            _error_table(summary),
            "",
            "## Latency Distribution Summary",
            _latency_table(rag_preds, agent_preds),
            "",
            "## Trace and Tooling Summary",
            _trace_table(rag_preds, agent_preds),
            "",
            "## Reproducibility Notes",
            "- Run data preparation and benchmark from this repository only.",
            "- Keep execution local or in private compute to avoid API exposure.",
            "- Share this report, the predictions JSON files, and source code in Git for peer review.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate proposal-ready benchmark artifacts")
    parser.add_argument("--summary", default="results/summary.json", help="Path to summary JSON")
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
        help="Directory to write report artifacts",
    )
    args = parser.parse_args()

    summary = _load_json(Path(args.summary))
    rag_preds = _load_json(Path(args.rag_predictions))
    agent_preds = _load_json(Path(args.agent_predictions))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_md = build_report(summary, rag_preds, agent_preds)
    (output_dir / "benchmark_report.md").write_text(report_md, encoding="utf-8")

    mermaid = _build_mermaid(summary)
    (output_dir / "benchmark_charts.mmd").write_text(mermaid, encoding="utf-8")

    index = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "summary": str(Path(args.summary)),
            "traditional_rag_predictions": str(Path(args.rag_predictions)),
            "agentic_multi_tool_predictions": str(Path(args.agent_predictions)),
        },
        "outputs": {
            "report_markdown": str(output_dir / "benchmark_report.md"),
            "mermaid_chart": str(output_dir / "benchmark_charts.mmd"),
        },
    }
    (output_dir / "artifact_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
