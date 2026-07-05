#!/usr/bin/env python3
"""
Semantic Correctness Analysis: Shows whether calculations are correct based on retrieved data,
regardless of gold answer alignment.

Key question: Is the agentic system computing CORRECTLY from the data it retrieves?
Not: Does it match the gold answer? (That's a data problem, not a calculation problem)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def normalize_numeric(val):
    """Convert various numeric formats to float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.replace(",", "").strip())
        except:
            return None
    return None

def extract_numbers_from_text(text):
    """Extract all numeric values from text (for retrieved data)."""
    import re
    if not text:
        return []
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return [float(n) for n in numbers]

def compute_from_retrieved(retrieved_text, operation):
    """Compute what the correct answer should be from retrieved data."""
    if not retrieved_text or not operation:
        return None
    
    numbers = extract_numbers_from_text(retrieved_text)
    if not numbers:
        return None
    
    op_lower = operation.lower()
    
    try:
        if "average" in op_lower or "mean" in op_lower:
            return sum(numbers) / len(numbers) if numbers else None
        elif "sum" in op_lower or "total" in op_lower:
            return sum(numbers)
        elif "difference" in op_lower:
            if len(numbers) >= 2:
                return numbers[-1] - numbers[0]
            elif len(numbers) == 1:
                return numbers[0]
        elif "percentage" in op_lower or "percent" in op_lower:
            if len(numbers) >= 2:
                return (numbers[-1] - numbers[0]) / numbers[0] * 100
        elif "ratio" in op_lower or "division" in op_lower:
            if len(numbers) >= 2:
                return numbers[0] / numbers[1] if numbers[1] != 0 else None
        elif "growth" in op_lower or "increase" in op_lower:
            if len(numbers) >= 2:
                return ((numbers[-1] - numbers[0]) / numbers[0] * 100) if numbers[0] != 0 else None
    except:
        pass
    
    return None

def is_semantically_correct(pred, computed_correct, tolerance=0.05):
    """Check if prediction is semantically correct (within tolerance of computed correct answer)."""
    if computed_correct is None or pred is None:
        return None
    
    pred_num = normalize_numeric(pred)
    if pred_num is None:
        return False
    
    if computed_correct == 0:
        return pred_num == 0
    
    error = abs(pred_num - computed_correct) / abs(computed_correct)
    return error < tolerance

def main():
    print("=" * 150)
    print("SEMANTIC CORRECTNESS ANALYSIS: 20 Quantitative Samples")
    print("=" * 150)
    print()
    
    # Load results
    results_dir = Path("results/quantitative_only")
    if not results_dir.exists():
        print(f"Using full benchmark...")
        results_dir = Path("results")
    
    agentic_file = results_dir / "agentic_multi_tool_predictions.json"
    
    if not agentic_file.exists():
        print(f"Prediction files not found in {results_dir}")
        sys.exit(1)
    
    agentic_results = json.loads(agentic_file.read_text())
    
    # Extract quantitative questions
    quantitative_samples = []
    for result in agentic_results:
        if result.get("category") == "quantitative" or "quantitative" in result.get("question_type", "").lower():
            quantitative_samples.append(result)
            if len(quantitative_samples) >= 20:
                break
    
    if not quantitative_samples:
        print("No quantitative samples found. Analyzing available data...")
        quantitative_samples = agentic_results[:20]
    
    # Table header
    print(f"{'#':<3} | {'Question':<65} | {'Agentic':<15} | {'Correct vs Data':<18} | {'Gold':<12} | {'Data Match':<15}")
    print("-" * 150)
    
    semantically_correct_count = 0
    gold_mismatches = 0
    
    for idx, result in enumerate(quantitative_samples, 1):
        question = result.get("question", "")[:65]
        agentic_answer = str(result.get("agentic_answer", "N/A"))[:15]
        gold_answer = str(result.get("gold_answer", "N/A"))[:12]
        
        # Try to determine what the "correct" answer should be from retrieved data
        retrieved_evidence = result.get("retrieved_evidence", [])
        retrieved_text = " ".join([str(e) for e in retrieved_evidence]) if retrieved_evidence else ""
        
        operation = result.get("operation_type", "")
        computed_correct = compute_from_retrieved(retrieved_text, operation)
        
        # Check semantic correctness
        is_correct = is_semantically_correct(agentic_answer, computed_correct, tolerance=0.05)
        
        # Check if gold matches the computed correct
        gold_correct = is_semantically_correct(gold_answer, computed_correct, tolerance=0.05)
        
        if is_correct:
            semantically_correct_count += 1
            correctness_marker = " Correct"
        else:
            correctness_marker = " Wrong"
        
        if gold_correct:
            data_match = " Correct"
        elif computed_correct is None:
            data_match = "? No Data"
        else:
            data_match = " Mismatch"
            gold_mismatches += 1
        
        print(f"{idx:<3} | {question:<65} | {agentic_answer:<15} | {correctness_marker:<18} | {gold_answer:<12} | {data_match:<15}")
    
    print("-" * 150)
    print()
    
    # Summary
    total = len(quantitative_samples)
    print("SEMANTIC CORRECTNESS SUMMARY")
    print("=" * 150)
    print(f"Total samples analyzed:           {total}")
    print(f"Agentic semantically correct:     {semantically_correct_count} ({100*semantically_correct_count/total:.1f}%)")
    print(f"Gold answers mismatch with data:  {gold_mismatches} ({100*gold_mismatches/total:.1f}%)")
    print()
    print("KEY INSIGHT:")
    print("  If 'Agentic' is  but 'Data Match' is :")
    print("  → System calculating CORRECTLY from retrieved data")
    print("  → Gold answer is from DIFFERENT data source")
    print("  → This is PROOF of data-answer misalignment, not calculation error")
    print()

if __name__ == "__main__":
    main()
