from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from gri_benchmark.runner import run_from_config
from gri_benchmark.settings import load_env_file


def _row(metrics: dict[str, float], key: str) -> float:
    return float(metrics.get(key, 0.0))


def _write_markdown(path: Path, summary: dict[str, dict[str, float]]) -> None:
    rag = summary["traditional_rag"]
    no_calc = summary["agentic_multi_tool_no_calculation"]
    no_ver = summary["agentic_multi_tool_no_verifier"]
    full = summary["agentic_multi_tool"]

    lines = [
        "# Agentic Tool Ablation Matrix",
        "",
        "| Pipeline | n_samples | exact_match | citation_precision | faithfulness | latency_ms | avg_tool_calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Traditional RAG | {_row(rag, 'n_samples'):.0f} | {_row(rag, 'exact_match'):.6f} | {_row(rag, 'citation_precision'):.6f} | {_row(rag, 'faithfulness'):.6f} | {_row(rag, 'latency_ms'):.3f} | {_row(rag, 'avg_tool_calls'):.2f} |",
        f"| Agentic w/o Calculation Tool | {_row(no_calc, 'n_samples'):.0f} | {_row(no_calc, 'exact_match'):.6f} | {_row(no_calc, 'citation_precision'):.6f} | {_row(no_calc, 'faithfulness'):.6f} | {_row(no_calc, 'latency_ms'):.3f} | {_row(no_calc, 'avg_tool_calls'):.2f} |",
        f"| Agentic w/o Verifier | {_row(no_ver, 'n_samples'):.0f} | {_row(no_ver, 'exact_match'):.6f} | {_row(no_ver, 'citation_precision'):.6f} | {_row(no_ver, 'faithfulness'):.6f} | {_row(no_ver, 'latency_ms'):.3f} | {_row(no_ver, 'avg_tool_calls'):.2f} |",
        f"| Full Agentic Multi-Tool | {_row(full, 'n_samples'):.0f} | {_row(full, 'exact_match'):.6f} | {_row(full, 'citation_precision'):.6f} | {_row(full, 'faithfulness'):.6f} | {_row(full, 'latency_ms'):.3f} | {_row(full, 'avg_tool_calls'):.2f} |",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agentic planner/tool/verifier ablations")
    parser.add_argument("--config", default="configs/benchmark.yaml", help="Base benchmark config")
    parser.add_argument("--env-file", default=".env", help="Environment file")
    parser.add_argument(
        "--output-json",
        default="results/ablation/agentic_tool_ablation_summary.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--output-md",
        default="docs/generated/agentic_tool_ablation_report.md",
        help="Output markdown report path",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    cfg["pipelines"] = [
        "traditional_rag",
        "agentic_multi_tool_no_calculation",
        "agentic_multi_tool_no_verifier",
        "agentic_multi_tool",
    ]
    cfg.setdefault("pipeline_options", {})
    cfg["pipeline_options"].setdefault("agentic_multi_tool_no_calculation", {"use_calculation_tool": False})
    cfg["pipeline_options"].setdefault("agentic_multi_tool_no_verifier", {"use_verifier": False})

    tmp_cfg = Path("results/ablation/.tmp_agentic_tool_ablation_config.yaml")
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    summary = run_from_config(tmp_cfg)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_markdown(Path(args.output_md), summary)

    print(
        json.dumps(
            {
                "pipelines": cfg["pipelines"],
                "output_json": str(out_json),
                "output_md": args.output_md,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
