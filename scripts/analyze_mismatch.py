#!/usr/bin/env python3
"""Detailed analysis of 10 failed quantitative examples"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gri_benchmark.data import load_examples
from gri_benchmark.pipelines.agentic_pipeline import AgenticMultiToolPipeline
from gri_benchmark.evidence import SimpleEvidenceRetriever

def normalize_numeric(val):
    """Convert value to float, handling commas and strings"""
    if val is None or val == "INSUFFICIENT_CONTEXT":
        return None
    try:
        return float(str(val).replace(",", ""))
    except:
        return None

def compute_tolerance_match(pred, gold, tolerance=0.05):
    """Check if |pred - gold| / |gold| < tolerance"""
    pred_num = normalize_numeric(pred)
    gold_num = normalize_numeric(gold)
    
    if pred_num is None or gold_num is None:
        return None
    
    if gold_num == 0:
        return pred_num == 0
    
    error = abs(pred_num - gold_num) / abs(gold_num)
    return error < tolerance, error

def main():
    # Load quantitative examples
    examples = load_examples("data/benchmark/one-table/gri-qa_quant.csv", split="single_table_quantitative")
    
    # Create pipeline and retriever
    retriever = SimpleEvidenceRetriever.from_jsonl("data/corpus/benchmark_corpus.jsonl")
    pipeline = AgenticMultiToolPipeline(strict_mode=True, retriever=retriever)
    
    print("=" * 100)
    print("QUANTITATIVE MISMATCH ANALYSIS: 10 Failed Examples")
    print("=" * 100)
    
    failed_count = 0
    analysis = []
    
    for i, example in enumerate(examples):
        if failed_count >= 10:
            break
        
        prediction = pipeline.answer(example)
        
        # Check if failed
        if str(prediction.answer) == str(example.gold_answer):
            continue  # Skip matches
        
        failed_count += 1
        
        # Get metadata
        metadata = prediction.metadata
        calc_trace = metadata.get("calculation_trace", {})
        retrieved_hits = metadata.get("retrieval_hits", [])
        
        # Compute tolerance metric
        tolerance_result = compute_tolerance_match(prediction.answer, example.gold_answer)
        
        pred_num = normalize_numeric(prediction.answer)
        gold_num = normalize_numeric(example.gold_answer)
        
        if tolerance_result and isinstance(tolerance_result, tuple):
            matches_tolerance, error = tolerance_result
        else:
            matches_tolerance = tolerance_result
            error = None
        
        # Categorize mismatch
        mismatch_type = "Unknown"
        mismatch_reason = ""
        
        calc_operation = calc_trace.get("operation", "None")
        calc_result = calc_trace.get("computed_result", "N/A")
        calc_confidence = calc_trace.get("confidence", 0)
        
        # Type A: Gold answer seems wrong or outdated
        if matches_tolerance:
            mismatch_type = "Type A (Tolerance Pass)"
            mismatch_reason = f"Within 5% tolerance - possibly outdated gold answer"
        
        # Type B: Calculation/Expectation mismatch
        elif calc_operation != "None" and calc_operation:
            if calc_confidence >= 0.85:
                mismatch_type = "Type B (Calc Mismatch)"
                mismatch_reason = f"High-conf calculation ({calc_operation}, conf={calc_confidence}) vs gold answer"
            else:
                mismatch_type = "Type B (Low Confidence)"
                mismatch_reason = f"Low-conf calculation ({calc_operation}, conf={calc_confidence})"
        
        # Type C: Unit/semantic mismatch
        elif retrieved_hits:
            units = set()
            for hit in retrieved_hits[:3]:
                if isinstance(hit, dict) and "units" in hit:
                    unit_val = hit.get("units", "unknown")
                    if isinstance(unit_val, list):
                        units.update(unit_val)
                    else:
                        units.add(str(unit_val))
            if len(units) > 1:
                mismatch_type = "Type C (Unit Mismatch)"
                mismatch_reason = f"Mixed units: {units}"
        
        # Type D: Insufficient context/no calculation
        if calc_operation == "None" or not calc_operation:
            if not retrieved_hits:
                mismatch_type = "Type D (No Evidence)"
                mismatch_reason = "No retrieved evidence for calculation"
        
        item = {
            "index": failed_count,
            "question": example.question,
            "prediction": prediction.answer,
            "gold_answer": example.gold_answer,
            "pred_numeric": pred_num,
            "gold_numeric": gold_num,
            "numeric_error": error,
            "tolerance_pass": matches_tolerance,
            "calc_operation": calc_operation,
            "calc_result": calc_result,
            "calc_confidence": calc_confidence,
            "retrieved_count": len(retrieved_hits),
            "mismatch_type": mismatch_type,
            "mismatch_reason": mismatch_reason,
        }
        analysis.append(item)
        
        # Print detailed analysis
        print(f"\n{'─' * 100}")
        print(f"[FAILED {failed_count}/10] {example.question[:70]}...")
        print(f"{'─' * 100}")
        print(f"  Your Answer:    {prediction.answer}")
        print(f"  Gold Answer:    {example.gold_answer}")
        print(f"  Numeric Diff:   {error*100:.1f}%" if error else "  Numeric Diff:   N/A")
        print(f"  5% Tolerance:   {' PASS' if matches_tolerance else ' FAIL'}")
        print(f"\n  Calculation:")
        print(f"    Operation:    {calc_operation}")
        print(f"    Result:       {calc_result}")
        print(f"    Confidence:   {calc_confidence}")
        print(f"  \n  Evidence:")
        print(f"    Retrieved:    {len(retrieved_hits)} hits")
        if retrieved_hits:
            for j, hit in enumerate(retrieved_hits[:2]):
                if isinstance(hit, dict):
                    print(f"      [{j}] {hit.get('record_id', 'unknown')}: {str(hit.get('value', 'N/A'))[:40]}")
        print(f"\n  Categorization: {mismatch_type}")
        print(f"  Reason:         {mismatch_reason}")
    
    # Summary statistics
    print(f"\n\n{'=' * 100}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 100}")
    
    print(f"\nTotal Failed Examples Analyzed: {len(analysis)}")
    
    # Categorization breakdown
    type_counts = {}
    for item in analysis:
        mtype = item["mismatch_type"]
        type_counts[mtype] = type_counts.get(mtype, 0) + 1
    
    print(f"\nMismatch Type Distribution:")
    for mtype, count in sorted(type_counts.items()):
        print(f"  {mtype}: {count}")
    
    # Tolerance metric
    tolerance_pass = sum(1 for item in analysis if item["tolerance_pass"])
    print(f"\nNumeric Tolerance Metric (5% tolerance):")
    print(f"  Pass: {tolerance_pass}/{len(analysis)} ({100*tolerance_pass/len(analysis):.1f}%)")
    print(f"  Fail: {len(analysis) - tolerance_pass}/{len(analysis)} ({100*(len(analysis)-tolerance_pass)/len(analysis):.1f}%)")
    
    # Average error
    errors = [item["numeric_error"] for item in analysis if item["numeric_error"] is not None]
    if errors:
        avg_error = sum(errors) / len(errors)
        print(f"\nAverage Numeric Error: {avg_error*100:.1f}%")
        print(f"Max Error: {max(errors)*100:.1f}%")
        print(f"Min Error: {min(errors)*100:.1f}%")
    
    # High-confidence calculation mismatch
    high_conf_calc = sum(1 for item in analysis if item["calc_confidence"] >= 0.85 and item["mismatch_type"].startswith("Type B"))
    print(f"\nHigh-Confidence Calculation Mismatches: {high_conf_calc}")
    
    # Save detailed analysis
    analysis_json = Path("results/quantitative_mismatch_analysis.json")
    analysis_json.write_text(json.dumps(analysis, indent=2))
    print(f"\nDetailed analysis saved to: {analysis_json}")

if __name__ == "__main__":
    main()
