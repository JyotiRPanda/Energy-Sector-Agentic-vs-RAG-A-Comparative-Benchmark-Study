from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from build_retrieval_corpus import build_corpus_from_config
from gri_benchmark.runner import run_from_config


METRICS = [
    "exact_match",
    "numeric_relative_error",
    "citation_precision",
    "faithfulness",
    "latency_ms",
]


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _load_name(config_path: Path) -> str:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data.get("strict_mode") and data.get("corpus_path"):
        return "strict_corpus"
    if data.get("strict_mode"):
        return "strict"
    return "non_strict"


def _ensure_corpus_if_needed(config_path: Path, rebuild: bool) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not data.get("strict_mode"):
        return
    corpus_path = data.get("corpus_path")
    if not corpus_path:
        return

    corpus_file = Path(corpus_path)
    if rebuild or not corpus_file.exists():
        build_corpus_from_config(config_path)


def _markdown_table(results: dict[str, dict[str, dict[str, float]]]) -> str:
    order = [k for k in ("non_strict", "strict_corpus", "strict") if k in results]
    lines: list[str] = ["# Ablation Comparison", ""]

    for pipeline_name in ("traditional_rag", "agentic_multi_tool"):
        lines.append(f"## {pipeline_name}")
        lines.append("")
        header = "| Metric | " + " | ".join(order) + " |"
        divider = "|---|" + "|".join(["---:" for _ in order]) + "|"
        lines.append(header)
        lines.append(divider)

        for metric in METRICS:
            values = []
            for cfg in order:
                values.append(_fmt(float(results[cfg][pipeline_name].get(metric, 0.0))))
            lines.append(f"| {metric} | " + " | ".join(values) + " |")

        if "non_strict" in order and "strict_corpus" in order:
            base = results["non_strict"][pipeline_name]
            strict = results["strict_corpus"][pipeline_name]
            lines.append("")
            lines.append("| Delta Metric (strict_corpus - non_strict) | Value |")
            lines.append("|---|---:|")
            for metric in METRICS:
                delta = float(strict.get(metric, 0.0)) - float(base.get(metric, 0.0))
                lines.append(f"| {metric} | {_fmt(delta)} |")

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark ablations and generate comparison tables")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/benchmark_ablation_non_strict.yaml",
            "configs/benchmark_ablation_strict_corpus.yaml",
        ],
        help="Benchmark config paths to run",
    )
    parser.add_argument(
        "--output-json",
        default="results/ablation/ablation_compare.json",
        help="Path to write ablation summary JSON",
    )
    parser.add_argument(
        "--output-md",
        default="docs/generated/ablation_comparison.md",
        help="Path to write ablation markdown table",
    )
    parser.add_argument(
        "--rebuild-corpus",
        action="store_true",
        help="Rebuild corpus for strict configs before running",
    )
    args = parser.parse_args()

    results: dict[str, dict[str, dict[str, float]]] = {}
    for cfg in args.configs:
        cfg_path = Path(cfg)
        _ensure_corpus_if_needed(cfg_path, rebuild=args.rebuild_corpus)
        label = _load_name(cfg_path)
        results[label] = run_from_config(cfg_path)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown_table(results), encoding="utf-8")

    print(
        json.dumps(
            {
                "configs": args.configs,
                "output_json": str(output_json),
                "output_md": str(output_md),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
