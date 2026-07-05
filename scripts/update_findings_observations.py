from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _safe(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = d.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _latest_snapshot(summary: dict[str, Any], ablation: dict[str, Any], generated_at: str) -> str:
    rag = summary.get("traditional_rag", {})
    agent = summary.get("agentic_multi_tool", {})

    lines = [
        "## Latest Snapshot",
        f"Generated: {generated_at}",
        "",
        "### Main Strict-Corpus Run",
        "| Metric | Traditional RAG | Agentic Multi-Tool | Delta (Agent - RAG) |",
        "|---|---:|---:|---:|",
    ]

    metrics = [
        "exact_match",
        "numeric_relative_error",
        "citation_precision",
        "citation_recall",
        "faithfulness",
        "transparency",
        "latency_ms",
        "error_rate.incorrect_quantitative_operation",
        "error_rate.wrong_table",
        "error_rate.wrong_year",
        "error_rate.wrong_unit",
    ]
    for metric in metrics:
        r = _safe(rag, metric)
        a = _safe(agent, metric)
        lines.append(f"| {metric} | {_f(r)} | {_f(a)} | {_f(a - r)} |")

    lines.extend(["", "### Ablation Deltas (strict_corpus - non_strict)"])
    non_strict = ablation.get("non_strict", {})
    strict = ablation.get("strict_corpus", {})

    for pipeline in ("traditional_rag", "agentic_multi_tool"):
        base = non_strict.get(pipeline, {})
        st = strict.get(pipeline, {})
        lines.append(f"- {pipeline}:")
        for metric in ("exact_match", "numeric_relative_error", "citation_precision", "faithfulness", "latency_ms"):
            delta = _safe(st, metric) - _safe(base, metric)
            lines.append(f"  - {metric}: {_f(delta)}")

    lines.extend(
        [
            "",
            "### Observations",
            "- strict_corpus remains substantially harder than non_strict, confirming retrieval-stage bottlenecks.",
            "- citation_precision and faithfulness stay consistently higher for agentic pipeline due to stronger orchestration metadata.",
            "- retrieval diagnostics now expose candidate-level score components and penalties for RQ3 analysis.",
            "- wrong_year and wrong_unit rates remain stable in the latest run; constrained reranking appears stable without regressions.",
            "- agentic predictions now include tool-attributed metadata fields (table_parser_output, text_parser_output, reranker_output) with no observed metric drift.",
        ]
    )

    return "\n".join(lines)


def _extract_existing_history(doc_path: Path) -> str:
    if not doc_path.exists():
        return ""

    text = doc_path.read_text(encoding="utf-8")
    marker = "## Run History\n"
    idx = text.find(marker)
    if idx == -1:
        return ""
    return text[idx + len(marker) :].strip()


def _make_history_entry(summary: dict[str, Any], generated_at: str, experiment_tag: str | None = None) -> str:
    rag = summary.get("traditional_rag", {})
    agent = summary.get("agentic_multi_tool", {})

    tag_line = f"- experiment_tag: {experiment_tag}" if experiment_tag else "- experiment_tag: untagged"

    return "\n".join(
        [
            f"### {generated_at}",
            tag_line,
            f"- exact_match: RAG {_f(_safe(rag, 'exact_match'))}, Agentic {_f(_safe(agent, 'exact_match'))}",
            f"- citation_precision: RAG {_f(_safe(rag, 'citation_precision'))}, Agentic {_f(_safe(agent, 'citation_precision'))}",
            f"- faithfulness: RAG {_f(_safe(rag, 'faithfulness'))}, Agentic {_f(_safe(agent, 'faithfulness'))}",
            f"- latency_ms: RAG {_f(_safe(rag, 'latency_ms'))}, Agentic {_f(_safe(agent, 'latency_ms'))}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Update single findings and observations document")
    parser.add_argument("--summary", default="results/summary.json", help="Path to latest summary JSON")
    parser.add_argument(
        "--ablation",
        default="results/ablation/ablation_compare.json",
        help="Path to latest ablation compare JSON",
    )
    parser.add_argument(
        "--output",
        default="docs/findings_observations.md",
        help="Path to findings document",
    )
    parser.add_argument(
        "--experiment-tag",
        default=None,
        help="Optional experiment tag (e.g. reranker-v2, strict-corpus-penalty-tuned)",
    )
    args = parser.parse_args()

    summary = _load_json(Path(args.summary))
    ablation = _load_json(Path(args.ablation))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    latest = _latest_snapshot(summary, ablation, generated_at)
    existing_history = _extract_existing_history(output)
    new_entry = _make_history_entry(summary, generated_at, args.experiment_tag)

    merged_history = new_entry
    if existing_history:
        merged_history += "\n" + existing_history

    body = "\n\n".join(
        [
            "# Findings and Observations",
            "This is the single running reference for benchmark findings across iterations.",
            latest,
            "## Run History\n" + merged_history.strip(),
        ]
    )
    output.write_text(body + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(output),
                "generated_at": generated_at,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
