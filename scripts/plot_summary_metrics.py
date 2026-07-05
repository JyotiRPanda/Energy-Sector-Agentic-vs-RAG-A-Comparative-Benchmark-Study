from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from gri_benchmark.data import load_examples
from gri_benchmark.evaluation.metrics import aggregate_metrics
from gri_benchmark.types import BenchmarkExample, Citation, Prediction


RESULTS_DIR = Path("results")

DEFAULT_DATASETS = [
    ("data/benchmark/one-table/gri-qa_extra.csv", "single_table_extractive"),
    ("data/benchmark/one-table/gri-qa_rel.csv", "single_table_relational"),
    ("data/benchmark/one-table/gri-qa_quant.csv", "single_table_quantitative"),
    ("data/benchmark/one-table/gri-qa_multistep.csv", "single_table_multistep"),
    ("data/benchmark/multi-table/gri-qa_multitable2-rel.csv", "multi_table_relational"),
    ("data/benchmark/multi-table/gri-qa_multitable2-quant.csv", "multi_table_quantitative"),
    ("data/benchmark/multi-table/gri-qa_multitable2-multistep.csv", "multi_table_multistep"),
]


def _read_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_result_dirs() -> Iterable[Path]:
    if not RESULTS_DIR.exists():
        return []
    dirs = [p for p in RESULTS_DIR.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def _artifact_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _candidate_dir_latest_artifact(run_dir: Path) -> float:
    summary = run_dir / "summary.json"
    agentic = run_dir / "agentic_multi_tool_predictions.json"
    trad = run_dir / "traditional_rag_predictions.json"
    return max(_artifact_mtime(summary), _artifact_mtime(agentic), _artifact_mtime(trad))


def _choose_run_dir(explicit_run_dir: Path | None = None) -> Path:
    if explicit_run_dir is not None:
        if not explicit_run_dir.exists() or not explicit_run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {explicit_run_dir}")
        return explicit_run_dir

    candidates = []
    for run_dir in _iter_result_dirs():
        latest = _candidate_dir_latest_artifact(run_dir)
        if latest > 0.0:
            candidates.append((latest, run_dir))

    if not candidates:
        raise FileNotFoundError(
            "No benchmark artifacts found. Expected one of: "
            "results/*/summary.json, results/*/agentic_multi_tool_predictions.json, "
            "results/*/traditional_rag_predictions.json"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _discover_sources(
    explicit_run_dir: Path | None = None,
) -> tuple[str, Path, Path | None, Path | None, Path | None]:
    run_dir = _choose_run_dir(explicit_run_dir=explicit_run_dir)

    # 1) Prefer aggregated summary for the chosen run.
    summary = run_dir / "summary.json"
    if summary.exists():
        return ("summary", run_dir, summary, None, None)

    # 2 + 3) Fallback to predictions for the same run.
    agentic = run_dir / "agentic_multi_tool_predictions.json"
    trad = run_dir / "traditional_rag_predictions.json"
    if agentic.exists() and trad.exists():
        return ("predictions", run_dir, None, agentic, trad)

    raise FileNotFoundError(
        "Chosen run directory is missing required artifacts. "
        f"run_dir={run_dir}; expected either summary.json or both prediction files."
    )


def _load_all_examples() -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for csv_path, split in DEFAULT_DATASETS:
        path = Path(csv_path)
        if path.exists():
            examples.extend(load_examples(path, split=split))
    if not examples:
        raise FileNotFoundError("Could not load any benchmark examples from default dataset paths")
    return examples


def _load_predictions(path: Path) -> list[Prediction]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    parsed: list[Prediction] = []
    for row in rows:
        citations = [Citation(**c) for c in row.get("citations", [])]
        parsed.append(
            Prediction(
                question_id=str(row.get("question_id", "")),
                pipeline_name=str(row.get("pipeline_name", "")),
                answer=str(row.get("answer", "")),
                latency_ms=float(row.get("latency_ms", 0.0) or 0.0),
                citations=citations,
                trace_steps=row.get("trace_steps", []) or [],
                metadata=row.get("metadata", {}) or {},
            )
        )
    return parsed


def _compute_summary_from_predictions(agentic_path: Path, trad_path: Path) -> dict:
    examples = _load_all_examples()
    agentic_preds = _load_predictions(agentic_path)
    trad_preds = _load_predictions(trad_path)
    return {
        "traditional_rag": aggregate_metrics(examples, trad_preds),
        "agentic_multi_tool": aggregate_metrics(examples, agentic_preds),
    }


def _to_percent(value: float) -> float:
    return value * 100.0


def _plot_two_bar(
    *,
    title: str,
    rag_label: str,
    rag_value: float,
    agentic_label: str,
    agentic_value: float,
    y_label: str,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    labels = [rag_label, agentic_label]
    values = [rag_value, agentic_value]
    colors = ["#4C78A8", "#F58518"]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_title(title, pad=12)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark metrics from latest results")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional specific run directory under results (e.g. results/full_nonlive_now)",
    )
    args = parser.parse_args()

    selected_run_dir = Path(args.run_dir) if args.run_dir else None
    source_type, run_dir, summary_path, agentic_path, trad_path = _discover_sources(
        explicit_run_dir=selected_run_dir
    )

    if source_type == "summary" and summary_path is not None:
        summary = _read_summary(summary_path)
        print(f"Using summary source: {summary_path}")
    else:
        assert agentic_path is not None and trad_path is not None
        summary = _compute_summary_from_predictions(agentic_path, trad_path)
        print(
            "Computed fallback metrics from predictions: "
            f"{agentic_path.name}, {trad_path.name}"
        )

    output_dir = run_dir / "plots"

    rag = summary["traditional_rag"]
    agentic = summary["agentic_multi_tool"]

    _plot_two_bar(
        title="Exact Match (higher is better)",
        rag_label="Traditional RAG",
        rag_value=_to_percent(rag["exact_match"]),
        agentic_label="Agentic",
        agentic_value=_to_percent(agentic["exact_match"]),
        y_label="Percent",
        output_file=output_dir / "exact_match.png",
    )

    _plot_two_bar(
        title="Numeric Relative Error (lower is better)",
        rag_label="Traditional RAG",
        rag_value=float(rag["numeric_relative_error"]),
        agentic_label="Agentic",
        agentic_value=float(agentic["numeric_relative_error"]),
        y_label="Error",
        output_file=output_dir / "numeric_error.png",
    )

    _plot_two_bar(
        title="Citation Precision (higher is better)",
        rag_label="Traditional RAG",
        rag_value=_to_percent(rag["citation_precision"]),
        agentic_label="Agentic",
        agentic_value=_to_percent(agentic["citation_precision"]),
        y_label="Percent",
        output_file=output_dir / "citation_precision.png",
    )

    _plot_two_bar(
        title="Faithfulness (higher is better)",
        rag_label="Traditional RAG",
        rag_value=_to_percent(rag["faithfulness"]),
        agentic_label="Agentic",
        agentic_value=_to_percent(agentic["faithfulness"]),
        y_label="Percent",
        output_file=output_dir / "faithfulness.png",
    )

    _plot_two_bar(
        title="Latency (ms, lower is better)",
        rag_label="Traditional RAG",
        rag_value=float(rag["latency_ms"]),
        agentic_label="Agentic",
        agentic_value=float(agentic["latency_ms"]),
        y_label="Milliseconds",
        output_file=output_dir / "latency.png",
    )

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
