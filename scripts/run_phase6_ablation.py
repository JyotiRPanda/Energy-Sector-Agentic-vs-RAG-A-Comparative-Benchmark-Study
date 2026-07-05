from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_retrieval_corpus import build_corpus_from_config
from gri_benchmark.runner import run_from_config


def _delta(a: float, b: float) -> float:
    return a - b


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-6 ablations and generate report")
    parser.add_argument(
        "--agentic-config",
        default="configs/benchmark_ablation_agentic_tools.yaml",
        help="Config for agentic full-vs-no-tools",
    )
    parser.add_argument(
        "--retrieval-config",
        default="configs/benchmark_ablation_retrieval_modes.yaml",
        help="Config for raw retrieval vs reranked retrieval",
    )
    parser.add_argument(
        "--output-json",
        default="results/ablation/phase6_ablation.json",
        help="Output path for phase6 ablation summary",
    )
    parser.add_argument(
        "--output-md",
        default="docs/generated/phase6_ablation_report.md",
        help="Output path for phase6 markdown report",
    )
    args = parser.parse_args()

    # Ensure strict corpora exist for both configs
    build_corpus_from_config(args.agentic_config)
    build_corpus_from_config(args.retrieval_config)

    agentic_results = run_from_config(args.agentic_config)
    retrieval_results = run_from_config(args.retrieval_config)

    full_agent = agentic_results["agentic_multi_tool"]
    no_tools = agentic_results["agentic_multi_tool_no_tools"]

    raw_ret = retrieval_results["traditional_rag_raw_retrieval"]
    reranked_ret = retrieval_results["traditional_rag_reranked"]

    summary = {
        "agentic_tools_ablation": {
            "agentic_multi_tool": full_agent,
            "agentic_multi_tool_no_tools": no_tools,
            "delta_full_minus_no_tools": {
                "exact_match": _delta(float(full_agent.get("exact_match", 0.0)), float(no_tools.get("exact_match", 0.0))),
                "citation_precision": _delta(
                    float(full_agent.get("citation_precision", 0.0)),
                    float(no_tools.get("citation_precision", 0.0)),
                ),
                "faithfulness": _delta(float(full_agent.get("faithfulness", 0.0)), float(no_tools.get("faithfulness", 0.0))),
                "latency_ms": _delta(float(full_agent.get("latency_ms", 0.0)), float(no_tools.get("latency_ms", 0.0))),
                "avg_tool_calls": _delta(
                    float(full_agent.get("avg_tool_calls", 0.0)),
                    float(no_tools.get("avg_tool_calls", 0.0)),
                ),
            },
        },
        "retrieval_rerank_ablation": {
            "traditional_rag_raw_retrieval": raw_ret,
            "traditional_rag_reranked": reranked_ret,
            "delta_reranked_minus_raw": {
                "exact_match": _delta(float(reranked_ret.get("exact_match", 0.0)), float(raw_ret.get("exact_match", 0.0))),
                "citation_precision": _delta(
                    float(reranked_ret.get("citation_precision", 0.0)),
                    float(raw_ret.get("citation_precision", 0.0)),
                ),
                "faithfulness": _delta(
                    float(reranked_ret.get("faithfulness", 0.0)),
                    float(raw_ret.get("faithfulness", 0.0)),
                ),
                "latency_ms": _delta(float(reranked_ret.get("latency_ms", 0.0)), float(raw_ret.get("latency_ms", 0.0))),
            },
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Phase 6 Ablation Report",
        "",
        "## A) Agentic Full vs Agentic No-Tools",
        "",
        "| Metric | Agentic Full | Agentic No-Tools | Delta (Full - No-Tools) |",
        "|---|---:|---:|---:|",
    ]

    for metric in ["exact_match", "citation_precision", "faithfulness", "latency_ms", "avg_tool_calls"]:
        a = float(full_agent.get(metric, 0.0))
        b = float(no_tools.get(metric, 0.0))
        lines.append(f"| {metric} | {a:.6f} | {b:.6f} | {a-b:+.6f} |")

    lines.extend(
        [
            "",
            "## B) Raw Retrieval vs Retrieval + Reranking",
            "",
            "| Metric | Raw Retrieval | Reranked Retrieval | Delta (Reranked - Raw) |",
            "|---|---:|---:|---:|",
        ]
    )

    for metric in ["exact_match", "citation_precision", "faithfulness", "latency_ms"]:
        raw = float(raw_ret.get(metric, 0.0))
        rr = float(reranked_ret.get(metric, 0.0))
        lines.append(f"| {metric} | {raw:.6f} | {rr:.6f} | {rr-raw:+.6f} |")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"output_json": str(out_json), "output_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
