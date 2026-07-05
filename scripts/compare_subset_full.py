from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = [
    "n_samples",
    "exact_match",
    "citation_precision",
    "citation_recall",
    "faithfulness",
    "transparency",
    "latency_ms",
    "error_rate.wrong_year",
    "error_rate.wrong_unit",
]


def _load(path: str) -> dict[str, dict[str, float]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _v(summary: dict[str, dict[str, float]], pipeline: str, metric: str) -> float:
    return float(summary.get(pipeline, {}).get(metric, 0.0))


def build_markdown(subset: dict[str, dict[str, float]], full: dict[str, dict[str, float]]) -> str:
    lines: list[str] = [
        "# Subset vs Full Benchmark Comparison",
        "",
        "| Pipeline | Metric | Subset | Full | Delta (Full - Subset) |",
        "|---|---|---:|---:|---:|",
    ]

    for pipeline in ("traditional_rag", "agentic_multi_tool"):
        for metric in METRICS:
            s = _v(subset, pipeline, metric)
            f = _v(full, pipeline, metric)
            d = f - s
            lines.append(f"| {pipeline} | {metric} | {s:.6f} | {f:.6f} | {d:+.6f} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare subset and full benchmark summaries")
    parser.add_argument("--subset", default="results/subset/summary.json")
    parser.add_argument("--full", default="results/full/summary.json")
    parser.add_argument("--output", default="docs/generated/subset_vs_full.md")
    args = parser.parse_args()

    subset = _load(args.subset)
    full = _load(args.full)
    md = build_markdown(subset, full)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    print(json.dumps({"output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
