#!/usr/bin/env python3
"""
Validation script: Run controlled_reasoning on 10 random questions
Log EXACTLY:
{
  "question": "...",
  "retrieved_chunks": "...",
  "operands": [...],
  "operation": "...",
  "predicted_answer": "...",
  "rag_original_answer": "...",
  "agentic_original_answer": "...",
  "gold_answer": "..."
}
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_check import controlled_reasoning


def load_predictions_list(filename: str):
    """Load predictions from JSON file (list format)."""
    pred_path = Path(__file__).parent.parent / "results" / filename
    
    if not pred_path.exists():
        print(f"  File not found at {pred_path}")
        return []
    
    with open(pred_path) as f:
        data = json.load(f)
        return data if isinstance(data, list) else []


def load_questions_with_text():
    """Load questions with text from re_evaluation_detailed."""
    eval_path = Path(__file__).parent.parent / "results" / "re_evaluation_detailed.json"
    
    if not eval_path.exists():
        print(f"  Evaluation file not found at {eval_path}")
        return {}
    
    with open(eval_path) as f:
        data = json.load(f)
        if isinstance(data, list):
            return {item.get('question_id'): item for item in data if isinstance(item, dict)}
    return {}


def validate_wrapper(num_samples: int = 10):
    """Run controlled_reasoning on random questions and log results."""
    
    # Load predictions from both pipelines
    rag_preds = load_predictions_list("traditional_rag_predictions.json")
    agentic_preds = load_predictions_list("agentic_multi_tool_predictions.json")
    
    # Load questions with text
    questions_with_text = load_questions_with_text()
    
    print(f" Loaded {len(rag_preds)} RAG predictions")
    print(f" Loaded {len(agentic_preds)} agentic predictions")
    print(f" Loaded {len(questions_with_text)} questions with text\n")
    
    # Create dict for easy lookup
    rag_dict = {p.get('question_id'): p for p in rag_preds if isinstance(p, dict)}
    agentic_dict = {p.get('question_id'): p for p in agentic_preds if isinstance(p, dict)}
    
    # Find common question IDs that have question text
    common_ids = set(rag_dict.keys()) & set(agentic_dict.keys()) & set(questions_with_text.keys())
    print(f" Found {len(common_ids)} questions with complete data\n")
    
    if len(common_ids) < num_samples:
        print(f"  Only {len(common_ids)} complete questions available")
        num_samples = len(common_ids)
    
    # Select random samples
    selected_ids = random.sample(list(common_ids), num_samples)
    
    results = []
    
    for i, qid in enumerate(selected_ids, 1):
        rag_pred = rag_dict[qid]
        agentic_pred = agentic_dict[qid]
        q_data = questions_with_text[qid]
        
        # Get actual question text
        question = q_data.get('question', qid)
        gold_answer = q_data.get('gold_answer')
        
        # Get original answers
        rag_answer = rag_pred.get('answer')
        agentic_answer = agentic_pred.get('answer')
        
        # Extract retrieved chunks from metadata retrieval_hits
        retrieved_chunks = []
        metadata = agentic_pred.get('metadata', {})
        retrieval_hits = metadata.get('retrieval_hits', [])
        
        for hit in retrieval_hits[:5]:  # Take first 5 hits
            if isinstance(hit, dict):
                # Construct evidence string with value + metadata
                value = hit.get('primary_value')
                years = hit.get('years', [])
                units = hit.get('units', [])
                
                if value:
                    # Format: "value (unit) from year(s)"
                    year_str = " ".join(str(y) for y in years) if years else "unknown year"
                    unit_str = f" {units[0]}" if units else ""
                    chunk = f"{value}{unit_str} in {year_str}"
                    retrieved_chunks.append(chunk)
        
        # Fallback: if no chunks, use question snippet
        if not retrieved_chunks:
            retrieved_chunks = [question[:80]]
        
        # Run wrapper
        try:
            result = controlled_reasoning(question, retrieved_chunks)
            predicted_answer = result.get('answer')
            operands = result.get('operands', [])
            operation = result.get('operation', 'unknown')
        except Exception as e:
            predicted_answer = None
            operands = []
            operation = f"ERROR: {str(e)}"
        
        # Convert numeric answers to float if possible
        try:
            rag_float = float(rag_answer) if rag_answer else None
        except (ValueError, TypeError):
            rag_float = None
        
        try:
            agentic_float = float(agentic_answer) if agentic_answer else None
        except (ValueError, TypeError):
            agentic_float = None
        
        try:
            gold_float = float(gold_answer) if gold_answer else None
        except (ValueError, TypeError):
            gold_float = None
        
        # Format output EXACTLY as requested
        log_entry = {
            "question": question[:200],  # First 200 chars
            "retrieved_chunks": "\n".join(retrieved_chunks),
            "operands": operands,
            "operation": operation,
            "predicted_answer": predicted_answer,
            "rag_original_answer": rag_float,
            "agentic_original_answer": agentic_float,
            "gold_answer": gold_float
        }
        
        results.append(log_entry)
        
        # Print formatted
        print(f"\n{'='*80}")
        print(f"Sample {i}/{num_samples}: {qid}")
        print(f"{'='*80}")
        print(json.dumps(log_entry, indent=2))
    
    return results


def main():
    print("🔍 WRAPPER CORRECTNESS VALIDATION")
    print("="*80)
    print("Running controlled_reasoning on 10 random questions...\n")
    
    # Validate wrapper
    results = validate_wrapper(num_samples=10)
    
    # Save results to file
    output_file = Path(__file__).parent.parent / "results" / "wrapper_validation.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f" Validation complete! Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print summary
    print("\n SUMMARY")
    print("-"*80)
    successful = sum(1 for r in results if r['predicted_answer'] is not None)
    print(f" Successful predictions: {successful}/{len(results)}")
    
    # Operations detected
    operations = {}
    for r in results:
        op = r['operation']
        operations[op] = operations.get(op, 0) + 1
    print(f"\n📋 Operations detected:")
    for op, count in sorted(operations.items()):
        print(f"   - {op}: {count}")
    
    # Check for silent failures
    errors = sum(1 for r in results if 'ERROR' in str(r['operation']))
    print(f"\n  Errors: {errors}/{len(results)}")
    
    if errors > 0:
        print("\n WRAPPER BREAKING SILENTLY - ERRORS DETECTED:")
        for i, r in enumerate(results, 1):
            if 'ERROR' in str(r['operation']):
                print(f"\n  Sample {i}: {r['question'][:50]}...")
                print(f"  Error: {r['operation']}")


if __name__ == "__main__":
    main()
