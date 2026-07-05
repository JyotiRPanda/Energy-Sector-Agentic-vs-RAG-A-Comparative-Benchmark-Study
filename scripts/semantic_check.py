#!/usr/bin/env python3
"""Semantic correctness check: Do calculated answers match actual data values in tables?"""

import sys
import json
import re
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gri_benchmark.data import load_examples
from gri_benchmark.evidence import SimpleEvidenceRetriever

def extract_numbers(text: str) -> List[float]:
    """
    Extract numeric values from raw retrieved text.
    
    Robustly extracts integers and floats from text, handling:
    - Negative numbers (e.g., "-42.5")
    - Comma-separated numbers (e.g., "12,500.50")
    - Mixed integer and float formats
    
    Args:
        text: Raw text containing numeric values
        
    Returns:
        List[float]: Extracted numbers in order of appearance, or empty list if none found
    """
    if not text:
        return []

    # Regex for ints and floats with optional commas
    # Matches: -123, 123.45, 12,500, -12,500.99, etc.
    pattern = r'-?\d[\d,]*\.?\d*'
    matches = re.findall(pattern, text)

    numbers = []
    for m in matches:
        try:
            value = float(m.replace(",", ""))
            numbers.append(value)
        except ValueError:
            continue

    return numbers


def extract_numeric_from_text(text):
    """
    Extract all numeric values from text (legacy wrapper).
    Deprecated: Use extract_numbers() instead.
    """
    return extract_numbers(text)




def is_derived_metric(chunk: str) -> bool:
    """
    Check if a chunk contains derived metrics (percentages, ratios, etc).
    Derived metrics should NOT be used as direct operands.
    
    Examples:
    - "efficiency: 92%" → derived (don't use %)
    - "intensity: 0.5 per ton" → derived (don't use 0.5)
    - "100 tons" → not derived (use it)
    """
    text = chunk.lower()
    
    return any(x in text for x in [
        "percent", "%", "intensity", "ratio",
        "efficiency", "per ", "index"
    ])


def filter_by_question_context(values: List[float], chunks: List[str], question: str) -> List[float]:
    """
    Filter operands based on overlap with question context.
    
    Strategy:
    - Extract key words from question
    - For each chunk, count word overlap with question
    - If overlap >= 2, include numbers from that chunk
    - Returns operands from contextually relevant chunks
    
    Example:
        >>> question = "What is total emissions from facility A?"
        >>> chunks = ["Facility A: 100 tons", "Facility B: 200 tons"]
        >>> filter_by_question_context([100, 200], chunks, question)
        [100]  # Only chunk with "Facility A" matches question context
    """
    q_words = set(question.lower().split())
    filtered = []

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(q_words.intersection(chunk_words))

        if overlap >= 2:  # simple threshold: at least 2 words match
            nums = extract_numbers(chunk)
            for n in nums:
                if n in values and n not in filtered:
                    filtered.append(n)

    return filtered


def rank_operands(question: str, retrieved_chunks: List[str], operation: str = "unknown") -> List[float]:
    """
    Final correct operand selection with flexible multi-value support.
    
    For GRI-QA, operands must satisfy:
    - Group numbers by unit/context (same metric)
    - Use ALL relevant values, not just 2-3
    - Prevent mixing incompatible units (%, tons, energy, etc.)
    - Allow up to 8 operands for flexible SUM/AVG operations
    
    Selection strategy:
    - Extract all numbers from all chunks
    - Filter out year tokens (2000-2030 range)
    - Group by detected unit/context (percentage, tons, energy, generic)
    - Select from dominant group (most values = most likely correct)
    - Deduplicate values
    - Return all relevant operands (up to 8)
    
    Args:
        question: The original question text (informational only)
        retrieved_chunks: List of retrieved text chunks containing candidate numbers
        
    Returns:
        List[float]: All relevant operands from same unit context (deduplicated, up to 8)
        
    Example:
        >>> chunks = ["Q1: 100 tons, Q2: 95 tons, Q3: 120 tons, Q4: 105 tons"]
        >>> rank_operands(question, chunks)
        [100.0, 95.0, 120.0, 105.0]  # All quarterly values selected
    """
    grouped = {}

    for chunk in retrieved_chunks:
        nums = extract_numbers(chunk)

        if not nums:
            continue

        text = chunk.lower()

        # Detect unit/context from chunk text
        if "%" in text or "percent" in text:
            key = "percentage"
        elif "ton" in text or "emission" in text:
            key = "tons"
        elif "mwh" in text or "energy" in text:
            key = "energy"
        else:
            key = "generic"

        # Filter out year-like values (metadata, not data)
        cleaned_nums = [n for n in nums if not (2000 <= n <= 2030)]

        if not cleaned_nums:
            continue

        # ❗ Skip derived metrics - EXCEPT for pct operations
        if is_derived_metric(chunk) and operation != "pct":
            continue

        grouped.setdefault(key, []).extend(cleaned_nums)

    if not grouped:
        return []

    # Choose dominant group (most values = most likely correct)
    best_group = max(grouped.items(), key=lambda x: len(x[1]))[1]

    # Deduplicate while preserving order
    seen = set()
    clean = []
    for n in best_group:
        if n not in seen:
            clean.append(n)
            seen.add(n)

    #  Filter by question context for more targeted operand selection
    filtered = filter_by_question_context(clean, retrieved_chunks, question)

    if len(filtered) >= 2:
        return filtered[:8]

    return clean[:8]  # fallback to all deduplicated values if context filter yields < 2


def detect_operation(question: str) -> str:
    """
    Detect the operation type required by a question.
    
    Analyzes question text to identify what calculation is being requested:
    - "average" or "mean" → "avg"
    - "sum" or "total" → "sum"
    - "difference", "reduction", "between", "increase" → "diff"
    - "percentage", "percent", "%" (WITHOUT diff keywords) → "pct"
    - None of above → "unknown"
    
    IMPORTANT: Checks diff keywords BEFORE pct keywords, so
    "percentage reduction" is detected as diff (not pct).
    
    Args:
        question: The question text to analyze
        
    Returns:
        str: One of ["sum", "avg", "diff", "pct", "unknown"]
        
    Example:
        >>> detect_operation("What is the average emissions?")
        'avg'
        >>> detect_operation("What is the total energy consumption?")
        'sum'
        >>> detect_operation("What is the percentage reduction?")
        'diff'
    """
    q = question.lower()
    
    # Check for average/mean FIRST (before diff/pct to avoid conflicts)
    if any(keyword in q for keyword in ["average", "mean", "avg"]):
        return "avg"
    
    # Check for sum/total
    if any(keyword in q for keyword in ["sum", "total", "combined"]):
        return "sum"
    
    # Check if percentage context exists
    pct_keywords = ["percentage", "percent", "%"]
    has_pct_keyword = any(keyword in q for keyword in pct_keywords)
    
    # Keywords that suggest relative/percentage change
    pct_change_keywords = ["change", "increase", "decrease"]
    
    # If we have percentage context WITH change keywords -> pct operation
    if has_pct_keyword and any(keyword in q for keyword in pct_change_keywords):
        return "pct"
    
    # Check for pure difference/reduction (without percentage context)
    diff_keywords = ["difference", "reduction", "between", "increase", "decrease", "change", "delta"]
    if any(keyword in q for keyword in diff_keywords):
        return "diff"
    
    # Check for standalone percentage
    if has_pct_keyword:
        return "pct"
    
    # No recognized operation
    return "unknown"


def compute_answer(operation: str, values: List[float]) -> float:
    """
    Safely compute answer using detected operation and selected operands.
    
    Handles all four operation types with proper error handling:
    - "sum": Adds all values
    - "avg": Computes arithmetic mean
    - "diff": Subtracts first from second (values[1] - values[0])
    - "pct": Computes percentage change ((v1-v0)/v0)*100
    
    Uses ONLY the top-2 operands selected by rank_operands().
    Prevents division by zero and handles edge cases gracefully.
    
    Args:
        operation: One of ["sum", "avg", "diff", "pct", "unknown"]
        values: List of numeric operands (typically top-2 from rank_operands)
        
    Returns:
        float: Calculated result, or None if operation fails
        
    Raises:
        ValueError: If operation is "unknown" or values list is empty
        
    Example:
        >>> compute_answer("avg", [100.0, 200.0])
        150.0
        >>> compute_answer("sum", [100.0, 200.0])
        300.0
        >>> compute_answer("diff", [100.0, 200.0])
        100.0
        >>> compute_answer("pct", [100.0, 150.0])
        50.0
    """
    if not values:
        raise ValueError("Cannot compute with empty values list")
    
    if operation == "unknown":
        raise ValueError("Cannot compute with unknown operation type")
    
    try:
        if operation == "sum":
            # Sum all values
            return float(sum(values))
        
        elif operation == "avg":
            # Arithmetic mean of all values
            return float(sum(values) / len(values))
        
        elif operation == "diff":
            # Percentage change: second - first
            if len(values) >= 2:
                return float(values[1] - values[0])
            else:
                # Single value: return as-is
                return float(values[0])
        
        elif operation == "pct":
            # Percentage change: ((new - old) / old) * 100
            if len(values) >= 2:
                if values[0] == 0:
                    # Division by zero: cannot compute percentage change
                    raise ValueError("Cannot compute percentage change with base value of 0")
                return float(((values[1] - values[0]) / values[0]) * 100)
            else:
                # Single value: cannot compute percentage change
                raise ValueError("Cannot compute percentage change with single value")
        
        else:
            # Should not reach here if operation is validated
            raise ValueError(f"Unknown operation type: {operation}")
    
    except (ValueError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f"Computation failed: {str(e)}")


def controlled_reasoning(question: str, retrieved_chunks: List[str]) -> dict:
    """
    Unified wrapper for semantic correctness validation across both RAG and agentic pipelines.
    
    This function orchestrates the complete semantic analysis pipeline:
    1. Rank operands by relevance to question
    2. Detect the operation type required
    3. Compute the result safely
    
    By using this wrapper, both RAG and agentic approaches use identical calculation logic,
    ensuring FAIR COMPARISON between the two retrieval strategies.
    
    Args:
        question: The original question text
        retrieved_chunks: List of text chunks retrieved by RAG or agentic system
        
    Returns:
        dict with keys:
            - "operation": Detected operation type ("sum", "avg", "diff", "pct", "unknown")
            - "operands": List of ranked operands used in calculation
            - "answer": Computed numeric result
            
    Raises:
        ValueError: If operation is unknown, operands empty, or computation fails
        
    Example:
        >>> question = "What was the average emissions?"
        >>> chunks = ["2023: 1500.5 tons", "2022: 1450.25 tons"]
        >>> result = controlled_reasoning(question, chunks)
        >>> result
        {
            'operation': 'avg',
            'operands': [1500.5, 1450.25],
            'answer': 1475.375
        }
    """
    try:
        # Step 1: Detect operation type from question (needed for operand filtering)
        operation = detect_operation(question)
        
        if operation == "unknown":
            raise ValueError(f"Could not detect operation type from question: {question}")
        
        # Step 2: Rank operands by relevance to question (using detected operation)
        operands = rank_operands(question, retrieved_chunks, operation)
        
        if not operands:
            raise ValueError("No relevant operands found in retrieved chunks")

        # Step 3: Compute result safely
        answer = compute_answer(operation, operands)
        
        return {
            "operation": operation,
            "operands": operands,
            "answer": answer
        }
    
    except ValueError as e:
        # Re-raise with context
        raise ValueError(f"Controlled reasoning failed: {str(e)}")


def main():
    # Load retriever to access corpus
    retriever = SimpleEvidenceRetriever.from_jsonl("data/corpus/benchmark_corpus.jsonl")
    
    # Load examples
    examples = load_examples("data/benchmark/one-table/gri-qa_quant.csv", split="single_table_quantitative")
    
    print("=" * 100)
    print("SEMANTIC CORRECTNESS CHECK: Verify Calculated Answers Against Actual Table Data")
    print("=" * 100)
    
    # Check first 10 examples
    checked = 0
    for i, example in enumerate(examples):
        if checked >= 10:
            break
        
        print(f"\n{'─' * 100}")
        print(f"[Q{checked+1}] {example.question[:80]}...")
        print(f"{'─' * 100}")
        print(f"  Gold Answer: {example.gold_answer}")
        
        # Retrieve evidence
        try:
            retrieved = retriever.retrieve(
                query=example.question,
                split=example.split,
                top_k=3
            )
        except:
            retrieved = []
        
        if not retrieved:
            print(f"   NO EVIDENCE RETRIEVED")
            checked += 1
            continue
        
        print(f"  Retrieved Evidence ({len(retrieved)} hits):")
        
        # Extract values from retrieved evidence
        all_values = []
        for j, hit in enumerate(retrieved[:3]):
            value = hit.record.primary_value
            text = hit.record.content_text
            
            print(f"    [{j}] ID: {hit.record.record_id}")
            print(f"        Value: {value}")
            print(f"        Text: {text[:100]}...")
            print(f"        Years: {hit.record.years}")
            print(f"        Units: {hit.record.units}")
            
            # Extract numbers from value and text
            try:
                if value:
                    numeric_val = float(str(value).replace(",", ""))
                    all_values.append(numeric_val)
            except:
                pass
        
        # Try to verify gold answer against extracted values
        if all_values:
            print(f"\n  Available Numeric Values: {all_values}")
            
            try:
                gold_num = float(str(example.gold_answer).replace(",", ""))
                
                # Check if gold answer is directly in the retrieved values
                if gold_num in all_values:
                    print(f"   Gold answer ({gold_num}) is directly in retrieved values!")
                    check_result = "DIRECT_MATCH"
                else:
                    # Check if gold could be average, sum, or diff
                    avg = sum(all_values) / len(all_values)
                    total = sum(all_values)
                    
                    if abs(avg - gold_num) / abs(gold_num) < 0.05:
                        print(f"   Gold answer ({gold_num}) matches AVERAGE of retrieved ({avg:.2f})")
                        check_result = "AVERAGE_MATCH"
                    elif abs(total - gold_num) / abs(gold_num) < 0.05:
                        print(f"   Gold answer ({gold_num}) matches SUM of retrieved ({total:.2f})")
                        check_result = "SUM_MATCH"
                    else:
                        print(f"   Gold answer ({gold_num}) doesn't match:")
                        print(f"      Average of retrieved: {avg:.2f}")
                        print(f"      Sum of retrieved: {total:.2f}")
                        check_result = "NO_MATCH"
            except:
                print(f"   Could not parse gold answer as numeric")
                check_result = "PARSE_ERROR"
        else:
            print(f"   No numeric values extracted from evidence")
            check_result = "NO_VALUES"
        
        print(f"  Semantic Check: {check_result}")
        checked += 1
    
    print(f"\n\n{'=' * 100}")
    print("INTERPRETATION")
    print(f"{'=' * 100}")
    print("""
If most checks show:
   DIRECT_MATCH / AVERAGE_MATCH / SUM_MATCH
  → Gold answers are CORRECT, calculated values should match these
  
   NO_MATCH
  → Corpus data differs from what gold answers expect
  → Possible: Different data version, partial rows, filtered subset
  
   NO_VALUES / PARSE_ERROR
  → Retrieved evidence structure doesn't contain expected values
  → Corpus may need re-indexing
    """)

if __name__ == "__main__":
    main()
