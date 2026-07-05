from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from gri_benchmark.data import load_examples
from gri_benchmark.evidence import SimpleEvidenceRetriever
from gri_benchmark.evaluation.error_taxonomy import summarize_errors
from gri_benchmark.evaluation.metrics import aggregate_metrics, citation_precision, exact_match
from gri_benchmark.live_clients import maybe_create_live_client
from gri_benchmark.pipelines.agentic_pipeline import AgenticMultiToolPipeline
from gri_benchmark.pipelines.rag_baseline import TraditionalRAGPipeline
from gri_benchmark.settings import load_env_file
from gri_benchmark.types import BenchmarkExample, Prediction

LEAKAGE_KEYS = {"value", "answer_value", "answer", "gold_answer", "label", "output"}


def _sanitize_examples_for_strict_mode(examples: list[BenchmarkExample]) -> list[BenchmarkExample]:
    sanitized: list[BenchmarkExample] = []
    for ex in examples:
        safe_md = {k: v for k, v in ex.metadata.items() if k not in LEAKAGE_KEYS}
        sanitized.append(
            BenchmarkExample(
                question_id=ex.question_id,
                question=ex.question,
                gold_answer=ex.gold_answer,
                split=ex.split,
                metadata=safe_md,
            )
        )
    return sanitized


def _to_prediction_dict(pred: Prediction) -> dict:
    return {
        "question_id": pred.question_id,
        "pipeline_name": pred.pipeline_name,
        "answer": pred.answer,
        "latency_ms": pred.latency_ms,
        "citations": [c.__dict__ for c in pred.citations],
        "trace_steps": pred.trace_steps,
        "metadata": pred.metadata,
    }


def _paired_outcome_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    accuracy = {"rag_win": 0, "agentic_win": 0, "tie": 0}
    citation = {"rag_win": 0, "agentic_win": 0, "tie": 0}

    for row in rows:
        rag_correct = bool(row["rag_correct"])
        agentic_correct = bool(row["agentic_correct"])
        rag_citation = bool(row["rag_citation_valid"])
        agentic_citation = bool(row["agentic_citation_valid"])

        if rag_correct and not agentic_correct:
            accuracy["rag_win"] += 1
        elif agentic_correct and not rag_correct:
            accuracy["agentic_win"] += 1
        else:
            accuracy["tie"] += 1

        if rag_citation and not agentic_citation:
            citation["rag_win"] += 1
        elif agentic_citation and not rag_citation:
            citation["agentic_win"] += 1
        else:
            citation["tie"] += 1

    return {"accuracy": accuracy, "citation_correctness": citation}


def _mcnemar_test(rows: list[dict], rag_key: str, agentic_key: str) -> dict[str, float | bool]:
    # b: rag true, agent false; c: rag false, agent true
    b = 0
    c = 0
    for row in rows:
        r = bool(row[rag_key])
        a = bool(row[agentic_key])
        if r and not a:
            b += 1
        elif a and not r:
            c += 1

    if (b + c) == 0:
        return {
            "b": float(b),
            "c": float(c),
            "chi_square": 0.0,
            "p_value": 1.0,
            "significant_0_05": False,
        }

    chi_square = ((abs(b - c) - 1) ** 2) / (b + c)
    # For 1 dof, p-value for chi-square can be computed via erfc.
    p_value = math.erfc(math.sqrt(chi_square / 2.0))
    return {
        "b": float(b),
        "c": float(c),
        "chi_square": chi_square,
        "p_value": p_value,
        "significant_0_05": p_value < 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired live benchmark (Ada retrieval + GPT-4o generation)")
    parser.add_argument("--config", default="configs/benchmark_full.yaml", help="Benchmark config path")
    parser.add_argument("--env-file", default=".env", help="Environment file path")
    parser.add_argument("--output-dir", default="results/live", help="Directory for live outputs")
    args = parser.parse_args()

    load_env_file(args.env_file)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    datasets = config["datasets"]
    strict_mode = bool(config.get("strict_mode", True))
    corpus_path = config.get("corpus_path")

    examples: list[BenchmarkExample] = []
    for ds in datasets:
        examples.extend(load_examples(ds["path"], split=ds.get("split", "eval")))

    retriever = None
    if strict_mode:
        if corpus_path:
            retriever = SimpleEvidenceRetriever.from_jsonl(corpus_path)
        else:
            retriever = SimpleEvidenceRetriever.from_examples(examples)

    prediction_examples = _sanitize_examples_for_strict_mode(examples) if strict_mode else examples

    live_client = maybe_create_live_client(force=True)
    if live_client is None:
        raise SystemExit(
            "Unable to initialize live Azure OpenAI client. "
            "Ensure PROJECT_ENDPOINT, API_KEY, MODEL_DEPLOYMENT, and EMBEDDING_DEPLOYMENT are set in .env."
        )

    rag = TraditionalRAGPipeline(strict_mode=strict_mode, retriever=retriever, live_client=live_client)
    agentic = AgenticMultiToolPipeline(strict_mode=strict_mode, retriever=retriever, live_client=live_client)

    rag_predictions: list[Prediction] = []
    agentic_predictions: list[Prediction] = []
    total = len(prediction_examples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, ex in enumerate(prediction_examples, start=1):
        rag_predictions.append(rag.answer(ex))
        agentic_predictions.append(agentic.answer(ex))

        # Emit periodic progress so long runs are observable and resumable artifacts exist.
        if idx % 50 == 0 or idx == total:
            print(f"[live-run] completed {idx}/{total} examples")
            (output_dir / "traditional_rag_live_predictions.partial.json").write_text(
                json.dumps([_to_prediction_dict(p) for p in rag_predictions], indent=2), encoding="utf-8"
            )
            (output_dir / "agentic_multi_tool_live_predictions.partial.json").write_text(
                json.dumps([_to_prediction_dict(p) for p in agentic_predictions], indent=2), encoding="utf-8"
            )

    rag_by_qid = {p.question_id: p for p in rag_predictions}
    agent_by_qid = {p.question_id: p for p in agentic_predictions}

    paired_rows: list[dict] = []
    for ex in examples:
        rag_pred = rag_by_qid[ex.question_id]
        agent_pred = agent_by_qid[ex.question_id]

        rag_correct = exact_match(ex.gold_answer, rag_pred.answer) > 0.999
        agentic_correct = exact_match(ex.gold_answer, agent_pred.answer) > 0.999

        rag_cit = (citation_precision(ex, rag_pred) or 0.0) >= 0.5
        agentic_cit = (citation_precision(ex, agent_pred) or 0.0) >= 0.5

        paired_rows.append(
            {
                "qid": ex.question_id,
                "rag_correct": rag_correct,
                "agentic_correct": agentic_correct,
                "rag_citation_valid": rag_cit,
                "agentic_citation_valid": agentic_cit,
                "rag_answer": rag_pred.answer,
                "agentic_answer": agent_pred.answer,
            }
        )

    paired_counts = _paired_outcome_counts(paired_rows)
    significance = {
        "accuracy_mcnemar": _mcnemar_test(paired_rows, "rag_correct", "agentic_correct"),
        "citation_mcnemar": _mcnemar_test(paired_rows, "rag_citation_valid", "agentic_citation_valid"),
    }

    rag_summary = {
        **aggregate_metrics(examples, rag_predictions),
        **{f"error_rate.{k}": v for k, v in summarize_errors(examples, rag_predictions).items()},
    }
    agentic_summary = {
        **aggregate_metrics(examples, agentic_predictions),
        **{f"error_rate.{k}": v for k, v in summarize_errors(examples, agentic_predictions).items()},
    }

    summary = {
        "traditional_rag": rag_summary,
        "agentic_multi_tool": agentic_summary,
        "paired_outcomes": paired_counts,
        "significance_tests": significance,
    }

    (output_dir / "live_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "live_predictions.json").write_text(json.dumps(paired_rows, indent=2), encoding="utf-8")
    (output_dir / "traditional_rag_live_predictions.json").write_text(
        json.dumps([_to_prediction_dict(p) for p in rag_predictions], indent=2), encoding="utf-8"
    )
    (output_dir / "agentic_multi_tool_live_predictions.json").write_text(
        json.dumps([_to_prediction_dict(p) for p in agentic_predictions], indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "sample_count": len(examples),
                "live_summary": str(output_dir / "live_summary.json"),
                "live_predictions": str(output_dir / "live_predictions.json"),
                "paired_outcomes": paired_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
