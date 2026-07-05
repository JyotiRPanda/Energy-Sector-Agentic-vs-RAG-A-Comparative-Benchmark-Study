from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from gri_benchmark.data import load_examples
from gri_benchmark.evaluation.error_taxonomy import classify_errors
from gri_benchmark.evaluation.metrics import aggregate_metrics
from gri_benchmark.types import BenchmarkExample, Citation, Prediction


ERROR_CATEGORIES = [
    "incorrect_quantitative_operation",
    "wrong_table",
    "wrong_unit",
]

PIPELINES = ["traditional_rag", "agentic_multi_tool"]


def _canonical_question_type(split: str) -> str:
    text = split.lower().strip()
    if "extractive" in text:
        return "extractive"
    if "relational" in text:
        return "relational"
    if "quantitative" in text:
        return "quantitative"
    if "multistep" in text:
        return "multistep"
    return text or "unknown"


def _load_examples_from_config(config_path: Path) -> list[BenchmarkExample]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    examples: list[BenchmarkExample] = []
    for ds in config["datasets"]:
        examples.extend(load_examples(ds["path"], split=ds.get("split", "eval")))
    return examples


def _load_predictions(path: Path) -> list[Prediction]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions: list[Prediction] = []
    for item in payload:
        citations = [Citation(**c) for c in item.get("citations", [])]
        predictions.append(
            Prediction(
                question_id=item["question_id"],
                pipeline_name=item["pipeline_name"],
                answer=item.get("answer", ""),
                latency_ms=float(item.get("latency_ms", 0.0) or 0.0),
                citations=citations,
                trace_steps=item.get("trace_steps", []),
                metadata=item.get("metadata", {}),
            )
        )
    return predictions


def _error_rates(
    examples: list[BenchmarkExample],
    predictions: list[Prediction],
    categories: list[str],
) -> dict[str, float]:
    ex_by_id = {e.question_id: e for e in examples}
    counts: Counter[str] = Counter()

    for pred in predictions:
        ex = ex_by_id[pred.question_id]
        counts.update(classify_errors(ex, pred))

    n = max(len(predictions), 1)
    return {name: counts.get(name, 0) / n for name in categories}


def _subset_by_qtype(
    examples: list[BenchmarkExample],
    predictions: list[Prediction],
) -> dict[str, tuple[list[BenchmarkExample], list[Prediction]]]:
    ex_by_id = {e.question_id: e for e in examples}
    grouped_examples: dict[str, list[BenchmarkExample]] = defaultdict(list)
    grouped_predictions: dict[str, list[Prediction]] = defaultdict(list)

    for pred in predictions:
        ex = ex_by_id[pred.question_id]
        qtype = _canonical_question_type(ex.split)
        grouped_examples[qtype].append(ex)
        grouped_predictions[qtype].append(pred)

    out: dict[str, tuple[list[BenchmarkExample], list[Prediction]]] = {}
    for qtype in sorted(grouped_predictions.keys()):
        out[qtype] = (grouped_examples[qtype], grouped_predictions[qtype])
    return out


def _build_markdown(report: dict) -> str:
    rows = report["error_category_breakdown"]["rows"]
    by_type = report["question_type_breakdown"]
    insights = report["insights"]

    lines: list[str] = [
        "# Phase 7 Deep Error Analysis",
        "",
        "## 1) Error Category Breakdown",
        "",
        "| Error Category | Traditional RAG | Agentic Multi-Tool | Delta (Agentic - RAG) |",
        "|---|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| {cat} | {rag:.6f} | {agentic:.6f} | {delta:+.6f} |".format(
                cat=row["error_category"],
                rag=row["traditional_rag"],
                agentic=row["agentic_multi_tool"],
                delta=row["delta_agentic_minus_rag"],
            )
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- Agentic delta on incorrect_quantitative_operation: {0:+.6f}".format(
                rows[0]["delta_agentic_minus_rag"] if rows else 0.0
            ),
            "- Agentic delta on wrong_table: {0:+.6f}".format(
                rows[1]["delta_agentic_minus_rag"] if len(rows) > 1 else 0.0
            ),
            "- Agentic delta on wrong_unit: {0:+.6f}".format(
                rows[2]["delta_agentic_minus_rag"] if len(rows) > 2 else 0.0
            ),
            "",
            "## 2) Breakdown By Question Type",
            "",
            "| Question Type | n | EM Traditional RAG | EM Agentic | Delta EM (Agentic - RAG) |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for qtype in ("extractive", "relational", "quantitative", "multistep"):
        block = by_type.get(qtype, {})
        if not block:
            continue
        lines.append(
            "| {qtype} | {n} | {rag:.6f} | {agentic:.6f} | {delta:+.6f} |".format(
                qtype=qtype,
                n=int(block["n_samples"]),
                rag=block["traditional_rag"]["exact_match"],
                agentic=block["agentic_multi_tool"]["exact_match"],
                delta=block["delta_agentic_minus_rag"]["exact_match"],
            )
        )

    lines.extend(
        [
            "",
            "Type insight summary:",
            "- Largest agentic gain is in: {0}".format(insights["largest_gain_type"]),
            "- Largest agentic drop is in: {0}".format(insights["largest_drop_type"]),
            "- Question types with positive agentic gain: {0}".format(
                ", ".join(insights["types_with_gain"]) if insights["types_with_gain"] else "none"
            ),
            "- Question types with no EM change: {0}".format(
                ", ".join(insights["types_with_no_change"]) if insights["types_with_no_change"] else "none"
            ),
            "- Agentic gain in multistep: {0:+.6f}".format(insights["multistep_delta_em"]),
            "- Agentic gain in extractive: {0:+.6f}".format(insights["extractive_delta_em"]),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 deep error analysis")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--pred-dir", default="results")
    parser.add_argument("--output-json", default="results/analysis/phase7_deep_error_analysis.json")
    parser.add_argument("--output-md", default="docs/generated/phase7_deep_error_analysis.md")
    args = parser.parse_args()

    config_path = Path(args.config)
    pred_dir = Path(args.pred_dir)

    examples = _load_examples_from_config(config_path)
    predictions = {
        pipeline: _load_predictions(pred_dir / f"{pipeline}_predictions.json") for pipeline in PIPELINES
    }

    ex_by_id = {e.question_id: e for e in examples}
    by_type_report: dict[str, dict] = {}

    for qtype in ("extractive", "relational", "quantitative", "multistep"):
        by_type_report[qtype] = {}

    error_rows = []
    overall_errors = {
        pipeline: _error_rates(examples, predictions[pipeline], ERROR_CATEGORIES) for pipeline in PIPELINES
    }

    for category in ERROR_CATEGORIES:
        rag_val = overall_errors["traditional_rag"][category]
        agentic_val = overall_errors["agentic_multi_tool"][category]
        error_rows.append(
            {
                "error_category": category,
                "traditional_rag": rag_val,
                "agentic_multi_tool": agentic_val,
                "delta_agentic_minus_rag": agentic_val - rag_val,
            }
        )

    for qtype in ("extractive", "relational", "quantitative", "multistep"):
        q_examples: list[BenchmarkExample] = []
        q_predictions: dict[str, list[Prediction]] = {"traditional_rag": [], "agentic_multi_tool": []}

        for qid, ex in ex_by_id.items():
            if _canonical_question_type(ex.split) == qtype:
                q_examples.append(ex)

        qid_set = {e.question_id for e in q_examples}
        for pipeline in PIPELINES:
            q_predictions[pipeline] = [p for p in predictions[pipeline] if p.question_id in qid_set]

        rag_metrics = aggregate_metrics(q_examples, q_predictions["traditional_rag"]) if q_examples else {}
        agentic_metrics = aggregate_metrics(q_examples, q_predictions["agentic_multi_tool"]) if q_examples else {}

        rag_errors = _error_rates(q_examples, q_predictions["traditional_rag"], ERROR_CATEGORIES) if q_examples else {}
        agentic_errors = _error_rates(q_examples, q_predictions["agentic_multi_tool"], ERROR_CATEGORIES) if q_examples else {}

        by_type_report[qtype] = {
            "n_samples": float(len(q_examples)),
            "traditional_rag": {
                "exact_match": rag_metrics.get("exact_match", 0.0),
                "citation_precision": rag_metrics.get("citation_precision", 0.0),
                "faithfulness": rag_metrics.get("faithfulness", 0.0),
                "error_rates": rag_errors,
            },
            "agentic_multi_tool": {
                "exact_match": agentic_metrics.get("exact_match", 0.0),
                "citation_precision": agentic_metrics.get("citation_precision", 0.0),
                "faithfulness": agentic_metrics.get("faithfulness", 0.0),
                "error_rates": agentic_errors,
            },
            "delta_agentic_minus_rag": {
                "exact_match": agentic_metrics.get("exact_match", 0.0) - rag_metrics.get("exact_match", 0.0),
                "citation_precision": agentic_metrics.get("citation_precision", 0.0)
                - rag_metrics.get("citation_precision", 0.0),
                "faithfulness": agentic_metrics.get("faithfulness", 0.0) - rag_metrics.get("faithfulness", 0.0),
                "error_rates": {
                    c: agentic_errors.get(c, 0.0) - rag_errors.get(c, 0.0) for c in ERROR_CATEGORIES
                },
            },
        }

    deltas = {
        qtype: by_type_report[qtype]["delta_agentic_minus_rag"]["exact_match"]
        for qtype in by_type_report
        if by_type_report[qtype]
    }
    if deltas:
        max_delta = max(deltas.values())
        min_delta = min(deltas.values())
        largest_gain_type = max(deltas, key=deltas.get) if max_delta > 0 else "none (no positive gain)"
        largest_drop_type = min(deltas, key=deltas.get) if min_delta < 0 else "none (no negative drop)"
    else:
        largest_gain_type = "n/a"
        largest_drop_type = "n/a"

    types_with_gain = sorted([k for k, v in deltas.items() if v > 0])
    types_with_no_change = sorted([k for k, v in deltas.items() if v == 0])

    report = {
        "error_category_breakdown": {
            "categories": ERROR_CATEGORIES,
            "rows": error_rows,
        },
        "question_type_breakdown": by_type_report,
        "insights": {
            "largest_gain_type": largest_gain_type,
            "largest_drop_type": largest_drop_type,
            "types_with_gain": types_with_gain,
            "types_with_no_change": types_with_no_change,
            "multistep_delta_em": deltas.get("multistep", 0.0),
            "extractive_delta_em": deltas.get("extractive", 0.0),
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_build_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(out_json),
                "output_md": str(out_md),
                "insights": report["insights"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()