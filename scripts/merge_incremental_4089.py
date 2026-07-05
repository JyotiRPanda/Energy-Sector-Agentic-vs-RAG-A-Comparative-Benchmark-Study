from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from gri_benchmark.data import load_examples
from gri_benchmark.evaluation.error_taxonomy import classify_errors
from gri_benchmark.evaluation.metrics import (
    citation_precision,
    citation_recall,
    exact_match,
    numeric_relative_error,
)
from gri_benchmark.types import BenchmarkExample, Citation, Prediction


ERR_IQO = "error_rate.incorrect_quantitative_operation"
ERR_MISCITATION = "error_rate.miscitation"
ERR_UNSUPPORTED = "error_rate.unsupported_claim"
ERR_WRONG_TABLE = "error_rate.wrong_table"
ERR_WRONG_UNIT = "error_rate.wrong_unit"


def _load_examples_from_config(config_path: Path) -> list[BenchmarkExample]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    examples: list[BenchmarkExample] = []
    for ds in config["datasets"]:
        examples.extend(load_examples(ds["path"], split=ds.get("split", "eval")))
    return examples


def _to_citation(obj: dict[str, Any]) -> Citation:
    return Citation(
        source_file=str(obj.get("source_file", "")),
        table_id=obj.get("table_id"),
        row_id=obj.get("row_id"),
        column_id=obj.get("column_id"),
        evidence_text=obj.get("evidence_text"),
        pdf_name=obj.get("pdf_name"),
        page_nbr=obj.get("page_nbr"),
        table_nbr=obj.get("table_nbr"),
        primary_value=obj.get("primary_value"),
        evidence_id=obj.get("evidence_id"),
        reason_used=obj.get("reason_used"),
    )


def _load_predictions(predictions_path: Path) -> list[Prediction]:
    rows = json.loads(predictions_path.read_text(encoding="utf-8"))
    preds: list[Prediction] = []
    for row in rows:
        preds.append(
            Prediction(
                question_id=str(row["question_id"]),
                pipeline_name=str(row["pipeline_name"]),
                answer=str(row.get("answer", "")),
                latency_ms=float(row.get("latency_ms", 0.0) or 0.0),
                citations=[_to_citation(c) for c in row.get("citations", [])],
                trace_steps=row.get("trace_steps", []),
                metadata=row.get("metadata", {}),
            )
        )
    return preds


def _family_from_split(split: str) -> str:
    if split.startswith(("multitable2", "multi_table")):
        return "multitable2"
    if split.startswith("multitable3"):
        return "multitable3"
    if split.startswith("multitable5"):
        return "multitable5"
    return "other"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _compute_metrics(examples: list[BenchmarkExample], predictions: list[Prediction]) -> dict[str, float]:
    by_id = {e.question_id: e for e in examples}
    em: list[float] = []
    nre: list[float] = []
    cp: list[float] = []
    cr: list[float] = []
    error_counts: dict[str, int] = defaultdict(int)

    for pred in predictions:
        ex = by_id[pred.question_id]
        em.append(exact_match(ex.gold_answer, pred.answer))

        nre_val = numeric_relative_error(ex.gold_answer, pred.answer)
        if nre_val is not None:
            nre.append(nre_val)

        cp_val = citation_precision(ex, pred)
        if cp_val is not None:
            cp.append(cp_val)

        cr_val = citation_recall(pred, ex)
        if cr_val is not None:
            cr.append(cr_val)

        for label in classify_errors(ex, pred):
            error_counts[label] += 1

    n = max(len(predictions), 1)
    out = {
        "n_samples": float(len(predictions)),
        "exact_match": _mean(em),
        "numeric_relative_error": _mean(nre),
        "citation_precision": _mean(cp),
        "citation_recall": _mean(cr),
    }
    for key, value in sorted(error_counts.items()):
        out[f"error_rate.{key}"] = value / n
    return out


def _subset_by_family(
    examples: list[BenchmarkExample],
    predictions: list[Prediction],
    family: str,
) -> tuple[list[BenchmarkExample], list[Prediction]]:
    allowed_ids = {e.question_id for e in examples if _family_from_split(e.split) == family}
    return [e for e in examples if e.question_id in allowed_ids], [p for p in predictions if p.question_id in allowed_ids]


def _delta(new: float, old: float) -> float:
    return new - old


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge baseline 3226 predictions with incremental 863 predictions")
    parser.add_argument("--baseline-config", default="configs/benchmark_full.yaml")
    parser.add_argument("--incremental-config", default="configs/benchmark_incremental_missing863.yaml")
    parser.add_argument("--baseline-dir", default="results/full")
    parser.add_argument("--incremental-dir", default="results/incremental_missing863")
    parser.add_argument("--output-dir", default="results/full_4089_incremental")
    args = parser.parse_args()

    baseline_examples = _load_examples_from_config(Path(args.baseline_config))
    incremental_examples = _load_examples_from_config(Path(args.incremental_config))
    merged_examples = baseline_examples + incremental_examples

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipelines = ["traditional_rag", "agentic_multi_tool"]
    merged_summary: dict[str, dict[str, float]] = {}
    comparison: dict[str, Any] = {"baseline_3226": {}, "merged_4089_incremental": {}, "delta": {}, "per_split_family": {}}

    for pipeline in pipelines:
        baseline_preds = _load_predictions(Path(args.baseline_dir) / f"{pipeline}_predictions.json")
        incremental_preds = _load_predictions(Path(args.incremental_dir) / f"{pipeline}_predictions.json")
        merged_preds = baseline_preds + incremental_preds

        baseline_metrics = _compute_metrics(baseline_examples, baseline_preds)
        merged_metrics = _compute_metrics(merged_examples, merged_preds)

        merged_summary[pipeline] = merged_metrics
        comparison["baseline_3226"][pipeline] = baseline_metrics
        comparison["merged_4089_incremental"][pipeline] = merged_metrics
        comparison["delta"][pipeline] = {
            "exact_match": _delta(merged_metrics.get("exact_match", 0.0), baseline_metrics.get("exact_match", 0.0)),
            "numeric_relative_error": _delta(
                merged_metrics.get("numeric_relative_error", 0.0),
                baseline_metrics.get("numeric_relative_error", 0.0),
            ),
            "citation_precision": _delta(
                merged_metrics.get("citation_precision", 0.0),
                baseline_metrics.get("citation_precision", 0.0),
            ),
            "citation_recall": _delta(
                merged_metrics.get("citation_recall", 0.0),
                baseline_metrics.get("citation_recall", 0.0),
            ),
            ERR_IQO: _delta(
                merged_metrics.get(ERR_IQO, 0.0),
                baseline_metrics.get(ERR_IQO, 0.0),
            ),
            ERR_MISCITATION: _delta(
                merged_metrics.get(ERR_MISCITATION, 0.0),
                baseline_metrics.get(ERR_MISCITATION, 0.0),
            ),
            ERR_UNSUPPORTED: _delta(
                merged_metrics.get(ERR_UNSUPPORTED, 0.0),
                baseline_metrics.get(ERR_UNSUPPORTED, 0.0),
            ),
            ERR_WRONG_TABLE: _delta(
                merged_metrics.get(ERR_WRONG_TABLE, 0.0),
                baseline_metrics.get(ERR_WRONG_TABLE, 0.0),
            ),
            ERR_WRONG_UNIT: _delta(
                merged_metrics.get(ERR_WRONG_UNIT, 0.0),
                baseline_metrics.get(ERR_WRONG_UNIT, 0.0),
            ),
        }

        family_stats: dict[str, dict[str, float]] = {}
        for family in ["multitable2", "multitable3", "multitable5"]:
            fam_examples, fam_preds = _subset_by_family(merged_examples, merged_preds, family)
            if fam_examples:
                family_stats[family] = _compute_metrics(fam_examples, fam_preds)
        comparison["per_split_family"][pipeline] = family_stats

    (output_dir / "summary.json").write_text(json.dumps(merged_summary, indent=2), encoding="utf-8")
    (output_dir / "comparison_3226_vs_4089_incremental.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# Incremental 4089 Comparison")
    lines.append("")
    lines.append("## Main Comparison (3226 baseline vs merged 4089 incremental)")
    lines.append("")
    lines.append("| Pipeline | Scope | n_samples | exact_match | numeric_relative_error | citation_precision | citation_recall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for pipeline in pipelines:
        b = comparison["baseline_3226"][pipeline]
        m = comparison["merged_4089_incremental"][pipeline]
        lines.append(
            f"| {pipeline} | 3226 | {int(b['n_samples'])} | {b['exact_match']:.4f} | {b['numeric_relative_error']:.4f} | {b['citation_precision']:.4f} | {b['citation_recall']:.4f} |"
        )
        lines.append(
            f"| {pipeline} | 4089 (merged) | {int(m['n_samples'])} | {m['exact_match']:.4f} | {m['numeric_relative_error']:.4f} | {m['citation_precision']:.4f} | {m['citation_recall']:.4f} |"
        )

    lines.append("")
    lines.append("## Delta (4089 merged - 3226 baseline)")
    lines.append("")
    lines.append("| Pipeline | Delta EM | Delta NRE | Delta C-Prec | Delta C-Rec | Delta IQO | Delta Miscitation | Delta Unsupported | Delta WrongTable | Delta WrongUnit |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for pipeline in pipelines:
        d = comparison["delta"][pipeline]
        lines.append(
            f"| {pipeline} | {d['exact_match']:+.4f} | {d['numeric_relative_error']:+.4f} | {d['citation_precision']:+.4f} | {d['citation_recall']:+.4f} | {d[ERR_IQO]:+.4f} | {d[ERR_MISCITATION]:+.4f} | {d[ERR_UNSUPPORTED]:+.4f} | {d[ERR_WRONG_TABLE]:+.4f} | {d[ERR_WRONG_UNIT]:+.4f} |"
        )

    lines.append("")
    lines.append("## Per-Split Family (multitable2 vs multitable3 vs multitable5)")
    lines.append("")
    lines.append("| Pipeline | Family | n_samples | exact_match | numeric_relative_error | citation_precision | citation_recall | incorrect_quantitative_operation | miscitation | unsupported_claim |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for pipeline in pipelines:
        for family in ["multitable2", "multitable3", "multitable5"]:
            row = comparison["per_split_family"][pipeline].get(family)
            if not row:
                continue
            lines.append(
                f"| {pipeline} | {family} | {int(row['n_samples'])} | {row['exact_match']:.4f} | {row['numeric_relative_error']:.4f} | {row['citation_precision']:.4f} | {row['citation_recall']:.4f} | {row.get(ERR_IQO, 0.0):.4f} | {row.get(ERR_MISCITATION, 0.0):.4f} | {row.get(ERR_UNSUPPORTED, 0.0):.4f} |"
            )

    (output_dir / "comparison_3226_vs_4089_incremental.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "summary": str(output_dir / "summary.json"),
        "comparison_json": str(output_dir / "comparison_3226_vs_4089_incremental.json"),
        "comparison_md": str(output_dir / "comparison_3226_vs_4089_incremental.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
