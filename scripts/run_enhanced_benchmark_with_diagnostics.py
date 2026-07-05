#!/usr/bin/env python3
"""
Script to run enhanced agentic benchmark and generate comprehensive diagnostics.

This script:
1. Runs the enhanced benchmark
2. Computes complex subset metrics
3. Generates oracle retrieval diagnostics
4. Produces comparison reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gri_benchmark.data import load_examples
from gri_benchmark.evaluation.metrics import aggregate_metrics
from gri_benchmark.evaluation.complex_subset import (
    compute_all_complex_subsets,
    format_complex_subset_report,
    compare_pipeline_subsets,
    score_predictions,
)
from gri_benchmark.evaluation.oracle_retrieval import (
    compute_oracle_retrieval_diagnostics,
    format_oracle_report,
)
from gri_benchmark.runner import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run enhanced agentic benchmark with diagnostics")
    parser.add_argument(
        "--config",
        default="configs/benchmark_agentic_enhanced.yaml",
        help="Path to benchmark config",
    )
    parser.add_argument(
        "--output-dir",
        default="results/enhanced",
        help="Directory for diagnostic outputs",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Also compute oracle retrieval diagnostics",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Environment file for credentials",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENHANCED AGENTIC BENCHMARK WITH DIAGNOSTICS")
    print("=" * 80)

    # Run benchmark
    print("\n[1/4] Running enhanced benchmark...")
    summary = run_from_config(args.config)
    print(f"✓ Benchmark complete. Summary saved to results/summary.json")

    # Load examples and predictions for diagnostics
    print("\n[2/4] Loading predictions for analysis...")
    config_path = Path(args.config)
    import yaml
    config = yaml.safe_load(config_path.read_text())
    
    examples = []
    for ds in config["datasets"]:
        examples.extend(load_examples(ds["path"], split=ds.get("split", "eval")))
    
    # Load predictions
    predictions_by_pipeline = {}
    scored_by_pipeline = {}
    
    for pipeline_name in config["pipelines"]:
        pred_file = Path("results") / f"{pipeline_name}_predictions.json"
        if not pred_file.exists():
            print(f"  ⚠ Predictions not found for {pipeline_name}")
            continue
        
        with open(pred_file) as f:
            preds_data = json.load(f)
        
        from gri_benchmark.types import Prediction, Citation
        predictions = []
        for p in preds_data:
            pred = Prediction(
                question_id=p["question_id"],
                pipeline_name=p["pipeline_name"],
                answer=p["answer"],
                latency_ms=p["latency_ms"],
                citations=[Citation(**c) for c in p["citations"]],
                trace_steps=p.get("trace_steps", []),
                metadata=p.get("metadata", {}),
            )
            predictions.append(pred)
        
        predictions_by_pipeline[pipeline_name] = predictions
        
        # Score predictions
        scored = score_predictions(examples, predictions)
        scored_by_pipeline[pipeline_name] = scored
        print(f"  ✓ Loaded {len(predictions)} predictions for {pipeline_name}")

    # Complex subset analysis
    print("\n[3/4] Computing complex subset metrics...")
    
    subset_results = {}
    for pipeline_name, scored_preds in scored_by_pipeline.items():
        subsets = compute_all_complex_subsets(examples, scored_preds)
        subset_results[pipeline_name] = subsets
        
        # Save subset report
        report = format_complex_subset_report(subsets)
        subset_file = output_dir / f"{pipeline_name}_complex_subsets.md"
        subset_file.write_text(report)
        print(f"  ✓ Complex subsets for {pipeline_name} saved to {subset_file.name}")

    # Comparison reports
    if len(subset_results) >= 2:
        pipeline_names = list(subset_results.keys())
        for i in range(len(pipeline_names) - 1):
            name1 = pipeline_names[i]
            name2 = pipeline_names[i + 1]
            
            comparison = compare_pipeline_subsets(
                subset_results[name1],
                subset_results[name2],
                name1,
                name2,
            )
            
            comp_file = output_dir / f"comparison_{name1}_vs_{name2}.md"
            comp_file.write_text(comparison)
            print(f"  ✓ Comparison {name1} vs {name2} saved to {comp_file.name}")

    # Oracle retrieval diagnostics
    if args.oracle:
        print("\n[4/4] Computing oracle retrieval diagnostics...")
        
        from gri_benchmark.evaluation.metrics import exact_match
        
        oracle_results = {}
        for pipeline_name, predictions in predictions_by_pipeline.items():
            # Create oracle predictions (gold answers)
            from gri_benchmark.types import Prediction, Citation
            oracle_preds = []
            for ex in examples:
                oracle_pred = Prediction(
                    question_id=ex.question_id,
                    pipeline_name=f"{pipeline_name}_oracle",
                    answer=ex.gold_answer,
                    latency_ms=0.1,
                    citations=[Citation(source_file=str(ex.metadata.get("source_file", "unknown")))],
                    trace_steps=[{"step": "oracle", "status": "ok", "details": "Direct gold answer"}],
                    metadata={"oracle_mode": True},
                )
                oracle_preds.append(oracle_pred)
            
            diagnostics = compute_oracle_retrieval_diagnostics(
                examples,
                predictions,
                oracle_preds,
                exact_match,
            )
            
            oracle_results[pipeline_name] = diagnostics
            
            report = format_oracle_report(diagnostics)
            oracle_file = output_dir / f"{pipeline_name}_oracle_diagnostics.md"
            oracle_file.write_text(report)
            print(f"  ✓ Oracle diagnostics for {pipeline_name} saved to {oracle_file.name}")
    else:
        print("\n[4/4] Skipping oracle retrieval diagnostics (use --oracle to enable)")

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files in: {output_dir.absolute()}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.md")):
        print(f"  - {f.name}")
    
    print("\nKey findings to review:")
    print("  1. Complex subset performance differences")
    print("  2. Which question types improved most")
    print("  3. Error label distributions by pipeline")
    if args.oracle:
        print("  4. Oracle retrieval potential improvements")


if __name__ == "__main__":
    main()
