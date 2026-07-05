from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_load_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.6f}"


def _extract_metrics(summary: dict, key: str) -> dict[str, float | None]:
    row = summary.get(key, {}) if isinstance(summary, dict) else {}
    if not isinstance(row, dict):
        row = {}
    return {
        "exact_match": row.get("exact_match"),
        "citation_precision": row.get("citation_precision"),
        "faithfulness": row.get("faithfulness"),
        "latency_ms": row.get("latency_ms"),
        "n_samples": row.get("n_samples"),
    }


def build_report(deterministic_summary: dict, live_summary: dict, phase6: dict) -> str:
    det_rag = _extract_metrics(deterministic_summary, "traditional_rag")
    det_agentic = _extract_metrics(deterministic_summary, "agentic_multi_tool")

    live_rag = _extract_metrics(live_summary, "traditional_rag")
    live_agentic = _extract_metrics(live_summary, "agentic_multi_tool")

    no_tools_src = phase6.get("agentic_tools_ablation", {}).get("agentic_multi_tool_no_tools", {})
    no_tools = {
        "exact_match": no_tools_src.get("exact_match"),
        "citation_precision": no_tools_src.get("citation_precision"),
        "faithfulness": no_tools_src.get("faithfulness"),
        "latency_ms": no_tools_src.get("latency_ms"),
        "n_samples": no_tools_src.get("n_samples"),
    }

    rows = [
        {
            "experiment": "Deterministic RAG",
            "purpose": "baseline",
            "metrics": det_rag,
            "source": "results/summary.json:traditional_rag",
            "status": "ready" if det_rag["n_samples"] is not None else "missing",
        },
        {
            "experiment": "Deterministic Agentic",
            "purpose": "architecture-only effect",
            "metrics": det_agentic,
            "source": "results/summary.json:agentic_multi_tool",
            "status": "ready" if det_agentic["n_samples"] is not None else "missing",
        },
        {
            "experiment": "Live RAG (Ada + GPT-4o)",
            "purpose": "realistic baseline",
            "metrics": live_rag,
            "source": "results/live/live_summary.json:traditional_rag",
            "status": "ready" if live_rag["n_samples"] is not None else "pending_live_run",
        },
        {
            "experiment": "Live Agentic (Ada + GPT-4o)",
            "purpose": "main contribution",
            "metrics": live_agentic,
            "source": "results/live/live_summary.json:agentic_multi_tool",
            "status": "ready" if live_agentic["n_samples"] is not None else "pending_live_run",
        },
        {
            "experiment": "Agentic (no tools)",
            "purpose": "ablation",
            "metrics": no_tools,
            "source": "results/ablation/phase6_ablation.json:agentic_tools_ablation.agentic_multi_tool_no_tools",
            "status": "ready" if no_tools["n_samples"] is not None else "missing",
        },
    ]

    lines = [
        "# Phase 8 Final Experiment Set",
        "",
        "| Experiment | Purpose | Status |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['purpose']} | {row['status']} |")

    lines.extend(
        [
            "",
            "## Metrics Snapshot",
            "",
            "| Experiment | n_samples | exact_match | citation_precision | faithfulness | latency_ms | Source |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        m = row["metrics"]
        lines.append(
            "| {experiment} | {n} | {em} | {cp} | {faith} | {lat} | {src} |".format(
                experiment=row["experiment"],
                n=_fmt(m["n_samples"]),
                em=_fmt(m["exact_match"]),
                cp=_fmt(m["citation_precision"]),
                faith=_fmt(m["faithfulness"]),
                lat=_fmt(m["latency_ms"]),
                src=row["source"],
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This table captures the final required experiment set and whether each run artifact is available.",
            "- Live rows require results/live/live_summary.json from scripts/run_live_paired_benchmark.py.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 8 final experiment set table")
    parser.add_argument("--deterministic-summary", default="results/summary.json")
    parser.add_argument("--live-summary", default="results/live/live_summary.json")
    parser.add_argument("--phase6", default="results/ablation/phase6_ablation.json")
    parser.add_argument("--output", default="docs/generated/phase8_final_experiment_set.md")
    args = parser.parse_args()

    deterministic_summary = _safe_load_json(Path(args.deterministic_summary))
    live_summary = _safe_load_json(Path(args.live_summary))
    phase6 = _safe_load_json(Path(args.phase6))

    report = build_report(deterministic_summary, live_summary, phase6)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print(json.dumps({"output": str(out)}, indent=2))


if __name__ == "__main__":
    main()