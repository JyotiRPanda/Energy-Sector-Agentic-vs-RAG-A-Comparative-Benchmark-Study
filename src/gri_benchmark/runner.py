from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import yaml

from gri_benchmark.data import load_examples
from gri_benchmark.evidence import SimpleEvidenceRetriever
from gri_benchmark.evaluation.error_taxonomy import summarize_errors
from gri_benchmark.evaluation.metrics import aggregate_metrics
from gri_benchmark.live_clients import maybe_create_live_client
from gri_benchmark.pipelines.agentic_pipeline import (
    AgenticMultiToolPipeline,
    AgenticNoCalculationToolPipeline,
    AgenticNoToolsPipeline,
    AgenticNoVerifierPipeline,
)
from gri_benchmark.pipelines.agentic_grounded_pipeline import AgenticGroundedPipeline
from gri_benchmark.pipelines.table_aware_agentic_pipeline import TableAwareAgenticPipeline
from gri_benchmark.pipelines.base import QAPipeline
from gri_benchmark.pipelines.enhanced_agentic_pipeline import EnhancedAgenticMultiToolPipeline
from gri_benchmark.pipelines.rag_baseline import (
    TraditionalRAGPipeline,
    TraditionalRAGRawRetrievalPipeline,
    TraditionalRAGRerankedPipeline,
)
from gri_benchmark.types import BenchmarkExample


PIPELINE_REGISTRY: dict[str, type[QAPipeline]] = {
    "traditional_rag": TraditionalRAGPipeline,
    "traditional_rag_reranked": TraditionalRAGRerankedPipeline,
    "traditional_rag_raw_retrieval": TraditionalRAGRawRetrievalPipeline,
    "agentic_multi_tool": AgenticMultiToolPipeline,
    "agentic_multi_tool_no_tools": AgenticNoToolsPipeline,
    "agentic_multi_tool_no_calculation": AgenticNoCalculationToolPipeline,
    "agentic_multi_tool_no_verifier": AgenticNoVerifierPipeline,
    "agentic_multi_tool_enhanced": EnhancedAgenticMultiToolPipeline,
    "agentic_multi_tool_grounded": AgenticGroundedPipeline,
    "agentic_table_aware": TableAwareAgenticPipeline,
}


LEAKAGE_KEYS = {
    "value",
    "answer_value",
    "answer",
    "gold_answer",
    "label",
    "output",
}


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


def run_from_config(config_path: str | Path, use_domain_aware_retrieval: bool = False) -> dict[str, dict[str, float]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    datasets = config["datasets"]
    pipeline_names = config["pipelines"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    strict_mode = bool(config.get("strict_mode", False))
    corpus_path = config.get("corpus_path")
    use_live_models = bool(config.get("use_live_models", False))
    use_live_env = str(os.getenv("USE_LIVE_MODELS", "")).strip().lower() in {"1", "true", "yes", "on"}
    use_live_requested = use_live_models or use_live_env
    pipeline_options = config.get("pipeline_options", {})
    domain_aware_retrieval = use_domain_aware_retrieval or bool(config.get("domain_aware_retrieval", False))

    examples = []
    for ds in datasets:
        examples.extend(load_examples(ds["path"], split=ds.get("split", "eval")))

    retriever = None
    if strict_mode:
        if corpus_path:
            retriever = SimpleEvidenceRetriever.from_jsonl(corpus_path)
        else:
            retriever = SimpleEvidenceRetriever.from_examples(examples)
        if domain_aware_retrieval:
            retriever.use_domain_aware = True
    prediction_examples = _sanitize_examples_for_strict_mode(examples) if strict_mode else examples

    # Locate annotation directory (for schema-aware structured retrieval)
    annotation_dir = config.get("annotation_dir")
    if annotation_dir is None:
        # Auto-discover relative to the config file
        config_parent = Path(config_path).parent.parent  # project root
        candidate = config_parent / "data" / "dataset" / "annotation"
        if candidate.is_dir():
            annotation_dir = str(candidate)

    summary: dict[str, dict[str, float]] = {}
    live_client = maybe_create_live_client(force=use_live_models)
    if use_live_requested and live_client is None:
        raise RuntimeError(
            "Live mode requested but Azure OpenAI client could not be initialized. "
            "Set PROJECT_ENDPOINT, API_KEY, MODEL_DEPLOYMENT, and EMBEDDING_DEPLOYMENT in .env "
            "(or disable use_live_models/USE_LIVE_MODELS)."
        )

    for pipeline_name in pipeline_names:
        pipeline_class = PIPELINE_REGISTRY[pipeline_name]
        options = pipeline_options.get(pipeline_name, {})
        # Pass annotation_dir to pipelines that support it (AgenticMultiToolPipeline variants)
        import inspect
        init_params = inspect.signature(pipeline_class.__init__).parameters
        if "annotation_dir" in init_params and annotation_dir:
            options = dict(options, annotation_dir=annotation_dir)
        pipeline = pipeline_class(strict_mode=strict_mode, retriever=retriever, live_client=live_client, **options)

        progress_interval = int(config.get("progress_interval", 25) or 25)
        predictions = []
        total = len(prediction_examples)
        run_start = perf_counter()
        for idx, example in enumerate(prediction_examples, start=1):
            predictions.append(pipeline.answer(example))
            if idx % progress_interval == 0 or idx == total:
                elapsed_s = perf_counter() - run_start
                print(
                    f"[benchmark] pipeline={pipeline_name} progress={idx}/{total} elapsed_s={elapsed_s:.1f}",
                    flush=True,
                )

        metrics = aggregate_metrics(examples, predictions)
        errors = summarize_errors(examples, predictions)
        report = {
            **metrics,
            **{f"error_rate.{k}": v for k, v in errors.items()},
        }
        summary[pipeline_name] = report

        def make_json_serializable(obj):
            """Convert non-serializable objects to serializable format."""
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
                # Object with __dict__ - convert to dict representation
                return make_json_serializable(obj.__dict__)
            else:
                return obj

        out_file = output_dir / f"{pipeline_name}_predictions.json"
        out_file.write_text(
            json.dumps(
                [
                    {
                        "question_id": p.question_id,
                        "pipeline_name": p.pipeline_name,
                        "answer": p.answer,
                        "latency_ms": p.latency_ms,
                        "citations": [c.__dict__ for c in p.citations],
                        "trace_steps": p.trace_steps,
                        "metadata": make_json_serializable(p.metadata),
                    }
                    for p in predictions
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[benchmark] pipeline={pipeline_name} wrote={out_file}", flush=True)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[benchmark] wrote_summary={output_dir / 'summary.json'}", flush=True)
    return summary
