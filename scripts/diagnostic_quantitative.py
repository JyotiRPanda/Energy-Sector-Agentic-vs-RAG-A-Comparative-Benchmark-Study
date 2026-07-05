#!/usr/bin/env python3
"""
Rapid diagnostic for quantitative questions specifically.
Checks: calculation_trace, retries, numeric values, different answers.
"""

import json
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gri_benchmark.data import load_examples
from gri_benchmark.runner import (
    PIPELINE_REGISTRY,
    maybe_create_live_client,
    _sanitize_examples_for_strict_mode,
)
from gri_benchmark.evidence import SimpleEvidenceRetriever
from gri_benchmark.settings import load_env_file
import yaml


def main():
    load_env_file(".env")
    
    print("╔" + "═" * 78 + "╗")
    print("║ QUANTITATIVE SUBSET DIAGNOSTIC: Check Calculation Engine")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Load config
    config_path = "configs/benchmark_agentic_enhanced.yaml"
    config = yaml.safe_load(Path(config_path).read_text())
    
    # Load only quantitative examples
    print("[1/4] Loading quantitative examples only...")
    all_examples = []
    for ds in config["datasets"]:
        all_examples.extend(load_examples(ds["path"], split=ds.get("split", "eval")))
    
    quant_examples = [ex for ex in all_examples if "quant" in ex.split.lower()][:10]
    print(f"  Loaded: {len(quant_examples)} quantitative examples")
    print()
    
    # Setup retriever
    print("[2/4] Setting up retriever...")
    corpus_path = config.get("corpus_path")
    retriever = SimpleEvidenceRetriever.from_jsonl(corpus_path) if corpus_path else SimpleEvidenceRetriever.from_examples(all_examples)
    print("  ✓ Ready")
    print()
    
    prediction_examples = _sanitize_examples_for_strict_mode(quant_examples)
    
    # Setup live client
    print("[3/4] Setting up live client...")
    live_client = maybe_create_live_client(force=True)
    print("  ✓ Ready")
    print()
    
    # Run pipelines
    print("[4/4] Running pipelines on quantitative subset...")
    print()
    
    for pipeline_name in ["traditional_rag", "agentic_multi_tool", "agentic_multi_tool_enhanced"]:
        print(f"\n{'─' * 80}")
        print(f"Pipeline: {pipeline_name}")
        print(f"{'─' * 80}\n")
        
        pipeline_class = PIPELINE_REGISTRY[pipeline_name]
        options = config.get("pipeline_options", {}).get(pipeline_name, {})
        pipeline = pipeline_class(strict_mode=True, retriever=retriever, live_client=live_client, **options)
        
        for i, example in enumerate(prediction_examples[:3]):
            pred = pipeline.answer(example)
            meta = pred.metadata or {}
            
            calc_trace = meta.get("calculation_trace")
            strategy = meta.get("agentic_strategy", "N/A")
            retry_count = meta.get("retry_count", 0)
            evidence_suff = meta.get("evidence_sufficiency", {})
            
            print(f"[Q{i+1}] {example.question_id}")
            print(f"  Question: {example.question[:60]}...")
            print(f"  Answer: {pred.answer[:50]}...")
            
            if calc_trace:
                print(f"   Calculation:")
                print(f"     Operation: {calc_trace.get('operation')}")
                print(f"     Input: {calc_trace.get('input_value')}")
                print(f"     Result: {calc_trace.get('computed_result')}")
                print(f"     Confidence: {calc_trace.get('confidence'):.2f}")
            else:
                print(f"   Calculation: ✗ (none)")
            
            print(f"  Strategy: {strategy}")
            print(f"  Retries: {retry_count}")
            print(f"  Evidence Sufficient: {evidence_suff.get('sufficient', 'N/A')}")
            
            if evidence_suff:
                print(f"  Coverage Issues: {len(evidence_suff.get('coverage_issues', []))}")
            print()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(" If you see:")
    print("   - calculation_trace with operation, input, result")
    print("   - Different answers between RAG and Agentic")
    print("   - evidence_sufficiency checks running")
    print()
    print("Then the calculation engine is working correctly!")


if __name__ == "__main__":
    main()
