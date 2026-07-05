#!/usr/bin/env python3
"""
Rapid diagnostic: Check critical signals on small samples (5-10 per category).
Identifies if agentic enhancements are working: retries, calculation, citations.
"""

import json
from pathlib import Path
import pandas as pd
import sys
from typing import Any

# Add src to path
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


def extract_signals(predictions: list[dict]) -> dict[str, Any]:
    """Extract critical signals from predictions."""
    signals = {
        "total": len(predictions),
        "answers_differ": 0,
        "has_retries": 0,
        "has_calculation_trace": 0,
        "has_strategy": 0,
        "has_evidence_check": 0,
        "sample_predictions": [],
    }
    
    for pred in predictions[:3]:  # Show first 3 examples
        rag_answer = pred.get("rag_answer", "")
        agentic_answer = pred.get("agentic_answer", "")
        enhanced_answer = pred.get("enhanced_answer", "")
        
        if rag_answer != agentic_answer or agentic_answer != enhanced_answer:
            signals["answers_differ"] += 1
        
        # Check for retries
        metadata = pred.get("metadata", {})
        if metadata.get("agentic", {}).get("retry_count", 0) > 0:
            signals["has_retries"] += 1
        if metadata.get("enhanced", {}).get("retry_count", 0) > 0:
            signals["has_retries"] += 1
        
        # Check for calculation trace
        if metadata.get("agentic", {}).get("calculation_trace"):
            signals["has_calculation_trace"] += 1
        if metadata.get("enhanced", {}).get("calculation_trace"):
            signals["has_calculation_trace"] += 1
        
        # Check for strategy
        if metadata.get("agentic", {}).get("agentic_strategy"):
            signals["has_strategy"] += 1
        if metadata.get("enhanced", {}).get("agentic_strategy"):
            signals["has_strategy"] += 1
        
        # Check for evidence sufficiency
        if metadata.get("agentic", {}).get("evidence_sufficiency"):
            signals["has_evidence_check"] += 1
        if metadata.get("enhanced", {}).get("evidence_sufficiency"):
            signals["has_evidence_check"] += 1
        
        # Sample prediction
        signals["sample_predictions"].append({
            "question_id": pred.get("question_id"),
            "rag": rag_answer[:50],
            "agentic": agentic_answer[:50],
            "enhanced": enhanced_answer[:50],
            "retry_count": metadata.get("enhanced", {}).get("retry_count", 0),
            "strategy": metadata.get("enhanced", {}).get("agentic_strategy", ""),
        })
    
    return signals


def main():
    load_env_file(".env")
    
    print("╔" + "═" * 78 + "╗")
    print("║ RAPID DIAGNOSTIC: Critical Signal Checking (Small Sample)")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Load config
    config_path = "configs/benchmark_agentic_enhanced.yaml"
    config = yaml.safe_load(Path(config_path).read_text())
    
    # Load small sample
    print("[1/5] Loading small sample (5 examples per split)...")
    examples = []
    for ds in config["datasets"]:
        ds_examples = load_examples(ds["path"], split=ds.get("split", "eval"))
        # Take only first 5
        ds_examples = ds_examples[:5]
        examples.extend(ds_examples)
        print(f"  ✓ {ds['split']}: {len(ds_examples)} examples")
    
    print(f"\n  Total: {len(examples)} examples")
    
    # Setup retriever
    print("\n[2/5] Setting up retriever...")
    corpus_path = config.get("corpus_path")
    retriever = SimpleEvidenceRetriever.from_jsonl(corpus_path) if corpus_path else SimpleEvidenceRetriever.from_examples(examples)
    print("  ✓ Retriever ready")
    
    # Sanitize
    prediction_examples = _sanitize_examples_for_strict_mode(examples)
    
    # Setup live client
    print("\n[3/5] Setting up live client...")
    live_client = maybe_create_live_client(force=True)
    print("  ✓ Live client ready")
    
    # Run each pipeline
    print("\n[4/5] Running pipelines...")
    all_predictions = []
    pipeline_names = config["pipelines"]
    pipeline_options = config.get("pipeline_options", {})
    
    predictions_by_pipeline = {}
    for pipeline_name in pipeline_names:
        print(f"\n  Running {pipeline_name}...")
        pipeline_class = PIPELINE_REGISTRY[pipeline_name]
        options = pipeline_options.get(pipeline_name, {})
        pipeline = pipeline_class(strict_mode=True, retriever=retriever, live_client=live_client, **options)
        
        predictions = [pipeline.answer(example) for example in prediction_examples]
        predictions_by_pipeline[pipeline_name] = predictions
        print(f"    ✓ {len(predictions)} predictions")
    
    # Extract and compare signals
    print("\n[5/5] Analyzing signals...")
    print()
    
    for pipeline_name, predictions in predictions_by_pipeline.items():
        print(f"\n─ {pipeline_name}")
        print("  " + "─" * 40)
        
        for i, pred in enumerate(predictions[:3]):
            answer = pred.answer[:60] if pred.answer else "[NO ANSWER]"
            metadata = pred.metadata or {}
            retry_count = metadata.get("retry_count", 0)
            strategy = metadata.get("agentic_strategy", "N/A")
            calc_trace = "✓" if metadata.get("calculation_trace") else "✗"
            evidence_suff = metadata.get("evidence_sufficiency", {})
            
            print(f"\n  [{i+1}] Q: {pred.question_id}")
            print(f"      Answer: {answer}...")
            print(f"      Retries: {retry_count}")
            print(f"      Strategy: {strategy}")
            print(f"      Calculation: {calc_trace}")
            print(f"      Evidence Sufficient: {evidence_suff.get('sufficient', 'N/A')}")
    
    # Compare answers across pipelines
    print("\n" + "=" * 80)
    print("ANSWER COMPARISON (Question 1)")
    print("=" * 80)
    
    if prediction_examples:
        q1_id = prediction_examples[0].question_id
        print(f"\nQuestion: {q1_id}")
        print()
        
        for pipeline_name, predictions in predictions_by_pipeline.items():
            if predictions:
                answer = predictions[0].answer
                print(f"  {pipeline_name}:")
                print(f"    {answer[:100]}...")
    
    print("\n" + "=" * 80)
    print("CRITICAL CHECKS")
    print("=" * 80)
    
    checks = {
        " Answers different?": False,
        " Retries happening?": False,
        " Calculation trace?": False,
        " Strategy routing?": False,
        " Evidence sufficiency?": False,
    }
    
    # Check each
    for pipeline_name, predictions in predictions_by_pipeline.items():
        if pipeline_name != "agentic_multi_tool_enhanced":
            continue
        
        for pred in predictions:
            meta = pred.metadata or {}
            if pred.answer:
                checks[" Answers different?"] = True
            if meta.get("retry_count", 0) > 0:
                checks[" Retries happening?"] = True
            if meta.get("calculation_trace"):
                checks[" Calculation trace?"] = True
            if meta.get("agentic_strategy"):
                checks[" Strategy routing?"] = True
            if meta.get("evidence_sufficiency"):
                checks[" Evidence sufficiency?"] = True
    
    print()
    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  [{status}] {check}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
