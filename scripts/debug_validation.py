#!/usr/bin/env python3
"""Debug script to check calculation validation on quantitative samples"""

import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gri_benchmark.data import load_examples
from gri_benchmark.pipelines.agentic_pipeline import AgenticMultiToolPipeline

def main():
    # Load just 20 quantitative samples
    examples = load_examples("data/benchmark/one-table/gri-qa_quant.csv", split="single_table_quantitative")
    examples = examples[:20]
    
    print(f"Testing {len(examples)} quantitative samples")
    print("=" * 80)
    
    # Get agentic pipeline
    pipeline = AgenticMultiToolPipeline(strict_mode=False)
    
    # Track validation results
    validation_stats = {
        "total": 0,
        "calculations_attempted": 0,
        "calculations_valid": 0,
        "calculations_invalid": 0,
        "reasons_for_rejection": {}
    }
    
    for i, example in enumerate(examples):
        validation_stats["total"] += 1
        
        try:
            # Call pipeline
            prediction = pipeline.answer(example)
            
            # Check metadata
            metadata = prediction.metadata
            calc_trace_used = metadata.get("calculation_trace_used", False)
            calculation_trace = metadata.get("calculation_trace", {})
            
            if calculation_trace:
                validation_stats["calculations_attempted"] += 1
                calc_result = calculation_trace.get("calculation_result", {})
                
                if calc_trace_used:
                    validation_stats["calculations_valid"] += 1
                    status = " USED"
                else:
                    validation_stats["calculations_invalid"] += 1
                    status = " REJECTED"
                    
                    # Log why it was rejected
                    success = calc_result.get("success", False)
                    confidence = calc_result.get("confidence", 0)
                    operation = calc_result.get("operation", "unknown")
                    computed = calc_result.get("computed_result", None)
                    
                    reason = ""
                    if not success:
                        reason = "not_successful"
                    elif confidence < 0.85:
                        reason = f"low_confidence ({confidence})"
                    else:
                        reason = "extreme_value"
                    
                    if reason not in validation_stats["reasons_for_rejection"]:
                        validation_stats["reasons_for_rejection"][reason] = 0
                    validation_stats["reasons_for_rejection"][reason] += 1
                
                print(f"\n[Q{i+1}] {example.question[:60]}...")
                print(f"  Answer: {str(prediction.answer)[:50]}...")
                print(f"  Operation: {operation}, Confidence: {confidence}")
                print(f"  Result: {computed}, Success: {success}")
                print(f"  Status: {status}")
        
        except Exception as e:
            print(f"\n[Q{i+1}] ERROR: {str(e)[:100]}")
            validation_stats["total"] += 1
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total samples: {validation_stats['total']}")
    print(f"Calculations attempted: {validation_stats['calculations_attempted']}")
    print(f"Calculations used (valid): {validation_stats['calculations_valid']}")
    print(f"Calculations rejected (invalid): {validation_stats['calculations_invalid']}")
    print(f"\nReasons for rejection:")
    for reason, count in validation_stats["reasons_for_rejection"].items():
        print(f"  - {reason}: {count}")

if __name__ == "__main__":
    main()
