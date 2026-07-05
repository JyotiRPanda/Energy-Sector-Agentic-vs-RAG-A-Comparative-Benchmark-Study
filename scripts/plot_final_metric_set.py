from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from gri_benchmark.data import load_examples
from gri_benchmark.evaluation.error_taxonomy import summarize_errors
from gri_benchmark.evaluation.metrics import exact_match
from gri_benchmark.types import BenchmarkExample, Citation, Prediction


DEFAULT_DATASETS = [
    ("data/benchmark/one-table/gri-qa_extra.csv", "single_table_extractive"),
    ("data/benchmark/one-table/gri-qa_rel.csv", "single_table_relational"),
    ("data/benchmark/one-table/gri-qa_quant.csv", "single_table_quantitative"),
    ("data/benchmark/one-table/gri-qa_multistep.csv", "single_table_multistep"),
    ("data/benchmark/multi-table/gri-qa_multitable2-rel.csv", "multi_table_relational"),
    ("data/benchmark/multi-table/gri-qa_multitable2-quant.csv", "multi_table_quantitative"),
    ("data/benchmark/multi-table/gri-qa_multitable2-multistep.csv", "multi_table_multistep"),
]


def _load_all_examples() -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for csv_path, split in DEFAULT_DATASETS:
        path = Path(csv_path)
        if path.exists():
            examples.extend(load_examples(path, split=split))
    if not examples:
        raise FileNotFoundError("Could not load benchmark examples from default dataset paths")
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


def _em_by_group(
    examples: list[BenchmarkExample],
    predictions: list[Prediction],
    group_filter,
) -> float:
    ex_by_id = {e.question_id: e for e in examples}
    values: list[float] = []
    for pred in predictions:
        ex = ex_by_id.get(pred.question_id)
        if ex is None:
            continue
        if group_filter(ex.split):
            values.append(exact_match(ex.gold_answer, pred.answer))
    return mean(values) if values else 0.0


def _task_wise_accuracy(examples: list[BenchmarkExample], rag: list[Prediction], agentic: list[Prediction]) -> dict[str, dict[str, float]]:
    filters = {
        "single_table_quantitative": lambda s: s == "single_table_quantitative",
        "single_table_multistep": lambda s: s == "single_table_multistep",
        "extractive": lambda s: s == "single_table_extractive",
        "multi_table splits": lambda s: s.startswith("multi_table_"),
    }
    out: dict[str, dict[str, float]] = {}
    for name, filt in filters.items():
        rag_em = _em_by_group(examples, rag, filt)
        agentic_em = _em_by_group(examples, agentic, filt)
        out[name] = {
            "traditional_rag": rag_em,
            "agentic_multi_tool": agentic_em,
            "delta_agentic_minus_rag": agentic_em - rag_em,
        }
    return out


def _multi_table_split_accuracy(
    examples: list[BenchmarkExample],
    rag: list[Prediction],
    agentic: list[Prediction],
) -> dict[str, dict[str, float]]:
    filters = {
        "multi_table_quantitative": lambda s: s == "multi_table_quantitative",
        "multi_table_relational": lambda s: s == "multi_table_relational",
        "multi_table_multistep": lambda s: s == "multi_table_multistep",
    }
    out: dict[str, dict[str, float]] = {}
    for name, filt in filters.items():
        rag_em = _em_by_group(examples, rag, filt)
        agentic_em = _em_by_group(examples, agentic, filt)
        out[name] = {
            "traditional_rag": rag_em,
            "agentic_multi_tool": agentic_em,
            "delta_agentic_minus_rag": agentic_em - rag_em,
        }
    return out


def _extractive_vs_quantitative_behavior(
    examples: list[BenchmarkExample],
    rag: list[Prediction],
    agentic: list[Prediction],
) -> dict[str, dict[str, float]]:
    filters = {
        "extractive": lambda s: s == "single_table_extractive",
        "quantitative": lambda s: s.endswith("_quantitative"),
    }
    out: dict[str, dict[str, float]] = {}
    for name, filt in filters.items():
        rag_em = _em_by_group(examples, rag, filt)
        agentic_em = _em_by_group(examples, agentic, filt)
        out[name] = {
            "traditional_rag": rag_em,
            "agentic_multi_tool": agentic_em,
            "delta_agentic_minus_rag": agentic_em - rag_em,
        }
    return out


def _error_distribution(examples: list[BenchmarkExample], rag: list[Prediction], agentic: list[Prediction]) -> dict[str, dict[str, float]]:
    rag_errors = summarize_errors(examples, rag)
    agentic_errors = summarize_errors(examples, agentic)

    mapping = {
        "incorrect_operation": "incorrect_quantitative_operation",
        "wrong_table": "wrong_table",
        "miscitation": "miscitation",
    }

    result: dict[str, dict[str, float]] = {}
    for public_name, internal_name in mapping.items():
        rag_rate = float(rag_errors.get(internal_name, 0.0))
        agentic_rate = float(agentic_errors.get(internal_name, 0.0))
        result[public_name] = {
            "traditional_rag": rag_rate,
            "agentic_multi_tool": agentic_rate,
            "delta_agentic_minus_rag": agentic_rate - rag_rate,
        }
    return result


def _plot_grouped_bar(
    labels: list[str],
    rag_values: list[float],
    agentic_values: list[float],
    *,
    title: str,
    y_label: str,
    output_file: Path,
    percent_axis: bool = True,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars_rag = ax.bar([i - width / 2 for i in x], rag_values, width=width, label="Traditional RAG", color="#4C78A8")
    bars_agentic = ax.bar([i + width / 2 for i in x], agentic_values, width=width, label="Agentic Multi-tool", color="#F58518")

    ax.set_title(title, pad=10)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if percent_axis:
        ax.set_ylim(0, max(100.0, max(rag_values + agentic_values) * 1.2))

    for bars in (bars_rag, bars_agentic):
        for bar in bars:
            val = bar.get_height()
            txt = f"{val:.1f}%" if percent_axis else f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, val, txt, ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def _plot_improvement(labels: list[str], deltas_pp: list[float], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#2CA02C" if v >= 0 else "#D62728" for v in deltas_pp]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(x, deltas_pp, color=colors)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_title("Improvement Over Baseline (Agentic - RAG)", pad=10)
    ax.set_ylabel("Percentage Points")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")

    for bar, value in zip(bars, deltas_pp):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.1f}pp", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def _plot_capability_gap(agentic_em: float, oracle_em: float, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    agentic_pct = agentic_em * 100.0
    oracle_pct = oracle_em * 100.0
    retrieval_gap = max(0.0, oracle_pct - agentic_pct)
    reasoning_gap = max(0.0, 100.0 - oracle_pct)

    labels = ["Agentic current", "Oracle ceiling", "Retrieval gap", "Reasoning gap"]
    values = [agentic_pct, oracle_pct, retrieval_gap, reasoning_gap]
    colors = ["#F58518", "#54A24B", "#E45756", "#72B7B2"]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Capability Gap: Retrieval vs Reasoning", pad=10)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final benchmark metric plots")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="results/full_nonlive_now",
        help="Benchmark result directory containing prediction files",
    )
    parser.add_argument(
        "--oracle-em",
        type=float,
        default=0.95,
        help="Oracle exact-match ceiling as fraction (default: 0.95)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rag_path = run_dir / "traditional_rag_predictions.json"
    agentic_path = run_dir / "agentic_multi_tool_predictions.json"

    if not rag_path.exists() or not agentic_path.exists():
        raise FileNotFoundError(
            f"Missing prediction files in {run_dir}. Expected {rag_path.name} and {agentic_path.name}."
        )

    examples = _load_all_examples()
    rag_preds = _load_predictions(rag_path)
    agentic_preds = _load_predictions(agentic_path)

    task_metrics = _task_wise_accuracy(examples, rag_preds, agentic_preds)
    multi_table_metrics = _multi_table_split_accuracy(examples, rag_preds, agentic_preds)
    behavior_metrics = _extractive_vs_quantitative_behavior(examples, rag_preds, agentic_preds)
    error_metrics = _error_distribution(examples, rag_preds, agentic_preds)

    overall_rag_em = _em_by_group(examples, rag_preds, lambda _: True)
    overall_agentic_em = _em_by_group(examples, agentic_preds, lambda _: True)

    labels_task = list(task_metrics.keys())
    rag_task = [task_metrics[k]["traditional_rag"] * 100.0 for k in labels_task]
    agentic_task = [task_metrics[k]["agentic_multi_tool"] * 100.0 for k in labels_task]

    labels_error = list(error_metrics.keys())
    rag_error = [error_metrics[k]["traditional_rag"] * 100.0 for k in labels_error]
    agentic_error = [error_metrics[k]["agentic_multi_tool"] * 100.0 for k in labels_error]

    output_dir = run_dir / "plots_final"

    _plot_grouped_bar(
        labels_task,
        rag_task,
        agentic_task,
        title="Task-wise Accuracy: Agentic Multi-tool vs Traditional RAG",
        y_label="Exact Match (%)",
        output_file=output_dir / "task_wise_accuracy.png",
    )

    labels_multi = list(multi_table_metrics.keys())
    rag_multi = [multi_table_metrics[k]["traditional_rag"] * 100.0 for k in labels_multi]
    agentic_multi = [multi_table_metrics[k]["agentic_multi_tool"] * 100.0 for k in labels_multi]

    _plot_grouped_bar(
        labels_multi,
        rag_multi,
        agentic_multi,
        title="Figure 5: Performance comparison on multi-table question answering tasks.",
        y_label="Exact Match (%)",
        output_file=output_dir / "figure5_multi_table_failure_analysis.png",
    )

    labels_behavior = list(behavior_metrics.keys())
    rag_behavior = [behavior_metrics[k]["traditional_rag"] * 100.0 for k in labels_behavior]
    agentic_behavior = [behavior_metrics[k]["agentic_multi_tool"] * 100.0 for k in labels_behavior]

    _plot_grouped_bar(
        labels_behavior,
        rag_behavior,
        agentic_behavior,
        title="Extractive vs Quantitative Behaviour",
        y_label="Exact Match (%)",
        output_file=output_dir / "extractive_vs_quantitative_behavior.png",
    )

    _plot_grouped_bar(
        labels_error,
        rag_error,
        agentic_error,
        title="Error Breakdown: Agentic Multi-tool vs Traditional RAG",
        y_label="Error Rate (%)",
        output_file=output_dir / "error_breakdown.png",
    )

    _plot_grouped_bar(
        ["Overall Exact Match"],
        [overall_rag_em * 100.0],
        [overall_agentic_em * 100.0],
        title="Exact Match (Overall)",
        y_label="Exact Match (%)",
        output_file=output_dir / "overall_exact_match.png",
    )

    split_deltas_pp = [
        task_metrics[k]["delta_agentic_minus_rag"] * 100.0 for k in labels_task
    ]
    overall_delta_pp = (overall_agentic_em - overall_rag_em) * 100.0

    _plot_improvement(
        labels_task + ["overall"],
        split_deltas_pp + [overall_delta_pp],
        output_file=output_dir / "improvement_over_baseline.png",
    )

    _plot_capability_gap(
        overall_agentic_em,
        args.oracle_em,
        output_file=output_dir / "capability_gap_oracle_vs_agentic.png",
    )

    summary = {
        "source_run_dir": str(run_dir),
        "overall": {
            "traditional_rag_exact_match": overall_rag_em,
            "agentic_multi_tool_exact_match": overall_agentic_em,
            "improvement_over_baseline_pp": overall_delta_pp,
            "oracle_exact_match_assumption": args.oracle_em,
        },
        "task_wise_accuracy": task_metrics,
        "multi_table_split_accuracy": multi_table_metrics,
        "extractive_vs_quantitative_behavior": behavior_metrics,
        "error_breakdown": error_metrics,
    }
    (output_dir / "final_metric_set.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved final metric plots to: {output_dir}")
    print(f"Overall EM (RAG): {overall_rag_em*100:.2f}%")
    print(f"Overall EM (Agentic): {overall_agentic_em*100:.2f}%")
    print(f"Improvement over baseline: {overall_delta_pp:+.2f}pp")


if __name__ == "__main__":
    main()
