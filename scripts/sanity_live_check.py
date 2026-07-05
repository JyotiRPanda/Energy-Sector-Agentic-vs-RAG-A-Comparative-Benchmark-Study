from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from gri_benchmark.data import load_examples
from gri_benchmark.evidence import SimpleEvidenceRetriever
from gri_benchmark.live_clients import maybe_create_live_client
from gri_benchmark.pipelines.agentic_pipeline import AgenticMultiToolPipeline
from gri_benchmark.pipelines.rag_baseline import TraditionalRAGPipeline
from gri_benchmark.settings import load_env_file
from gri_benchmark.types import BenchmarkExample


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


def _ratio(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return part / whole


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check live Ada retrieval + GPT-4o generation")
    parser.add_argument("--config", default="configs/benchmark_full.yaml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--sample-size", type=int, default=15)
    args = parser.parse_args()

    load_env_file(args.env_file)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    examples: list[BenchmarkExample] = []
    for ds in config["datasets"]:
        examples.extend(load_examples(ds["path"], split=ds.get("split", "eval")))

    sample_size = max(1, min(args.sample_size, len(examples)))
    examples = examples[:sample_size]

    strict_mode = bool(config.get("strict_mode", True))
    corpus_path = config.get("corpus_path")

    if corpus_path:
        retriever = SimpleEvidenceRetriever.from_jsonl(corpus_path)
    else:
        retriever = SimpleEvidenceRetriever.from_examples(examples)

    prediction_examples = _sanitize_examples_for_strict_mode(examples) if strict_mode else examples

    live_client = maybe_create_live_client(force=True)
    if live_client is None:
        raise SystemExit("Live client initialization failed. Check PROJECT_ENDPOINT/API_KEY/MODEL_DEPLOYMENT/EMBEDDING_DEPLOYMENT in .env.")

    rag = TraditionalRAGPipeline(strict_mode=strict_mode, retriever=retriever, live_client=live_client)
    agent = AgenticMultiToolPipeline(strict_mode=strict_mode, retriever=retriever, live_client=live_client)

    rag_preds = [rag.answer(ex) for ex in prediction_examples]
    agent_preds = [agent.answer(ex) for ex in prediction_examples]

    rag_relevant = 0
    rag_grounded = 0
    for ex, pred in zip(examples, rag_preds):
        expected_src = str(ex.metadata.get("source_file", "")).strip()
        hits = pred.metadata.get("retrieval_hits", [])
        top_src = str(hits[0].get("source_file", "")).strip() if hits else ""
        if expected_src and top_src and expected_src == top_src:
            rag_relevant += 1

        if pred.answer != "INSUFFICIENT_CONTEXT" and pred.citations:
            rag_grounded += 1

    agent_relevant = 0
    agent_grounded = 0
    agent_tool_calls = 0
    agent_has_trace = 0
    agent_has_citations = 0
    for ex, pred in zip(examples, agent_preds):
        expected_src = str(ex.metadata.get("source_file", "")).strip()
        hits = pred.metadata.get("retrieval_hits", [])
        top_src = str(hits[0].get("source_file", "")).strip() if hits else ""
        if expected_src and top_src and expected_src == top_src:
            agent_relevant += 1

        if pred.answer != "INSUFFICIENT_CONTEXT" and pred.citations:
            agent_grounded += 1

        trace_steps = pred.trace_steps or []
        if trace_steps:
            agent_has_trace += 1
        if any(str(step.get("step", "")).startswith("tool.") for step in trace_steps):
            agent_tool_calls += 1
        if pred.citations:
            agent_has_citations += 1

    report = {
        "sample_size": sample_size,
        "rag": {
            "retrieval_relevance_ratio": round(_ratio(rag_relevant, sample_size), 4),
            "grounded_answer_ratio": round(_ratio(rag_grounded, sample_size), 4),
        },
        "agentic": {
            "retrieval_relevance_ratio": round(_ratio(agent_relevant, sample_size), 4),
            "grounded_answer_ratio": round(_ratio(agent_grounded, sample_size), 4),
            "tool_call_ratio": round(_ratio(agent_tool_calls, sample_size), 4),
            "trace_log_ratio": round(_ratio(agent_has_trace, sample_size), 4),
            "citation_ratio": round(_ratio(agent_has_citations, sample_size), 4),
        },
        "checks": {
            "retrieval_not_random": _ratio(agent_relevant, sample_size) >= 0.6 and _ratio(rag_relevant, sample_size) >= 0.6,
            "grounded_outputs": _ratio(agent_grounded, sample_size) >= 0.8 and _ratio(rag_grounded, sample_size) >= 0.8,
            "agent_calls_tools": _ratio(agent_tool_calls, sample_size) >= 0.8,
            "agent_has_trace_logs": _ratio(agent_has_trace, sample_size) >= 0.8,
            "agent_has_citations": _ratio(agent_has_citations, sample_size) >= 0.8,
        },
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
