#!/usr/bin/env python3
"""
Full Benchmark Script - Compare Traditional RAG vs Agentic vs Controlled Reasoning

Compares three approaches across 2750 quantitative GRI-QA questions:
1. Traditional RAG (keyword-based retrieval + direct extraction)
2. Agentic (multi-tool agent with reasoning)
3. Controlled Reasoning (fair comparison wrapper with 3-layer semantic filtering)
"""

import json
import math
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

#  Import semantic correctness wrapper
from semantic_check import controlled_reasoning


#  HELPERS
def safe_float(x):
    """Safely convert value to float."""
    try:
        if isinstance(x, str):
            x = x.strip()
            if x.lower() in ['nan', 'none', '']:
                return None
        return float(x)
    except (ValueError, TypeError):
        return None


def exact_match(pred: Optional[float], gold: Optional[float]) -> int:
    """Check exact match: absolute difference < 1.0."""
    if pred is None or gold is None:
        return 0
    return 1 if abs(pred - gold) < 1.0 else 0


def tol_20(pred: Optional[float], gold: Optional[float]) -> int:
    """Check tolerance match: within 20% relative error."""
    if pred is None or gold is None:
        return 0
    if gold == 0:
        return 1 if pred == 0 else 0
    return 1 if abs(pred - gold) / abs(gold) < 0.2 else 0


def tol_50(pred: Optional[float], gold: Optional[float]) -> int:
    """Check tolerance match: within 50% relative error."""
    if pred is None or gold is None:
        return 0
    if gold == 0:
        return 1 if pred == 0 else 0
    return 1 if abs(pred - gold) / abs(gold) < 0.5 else 0


def detect_question_type(q: str) -> str:
    """Classify question into type: extractive, quant, derived."""
    q = q.lower()
    
    if any(x in q for x in ["sum", "total", "increase", "difference", "change"]):
        return "quant"
    if any(x in q for x in ["percent", "%", "reduction", "growth", "ratio", "intensity"]):
        return "derived"
    
    return "extractive"


def detect_operation(q: str) -> str:
    """Detect operation type from question: sum|avg|diff|pct|unknown."""
    q = q.lower()
    
    if "average" in q or "mean" in q:
        return "avg"
    if any(x in q for x in ["sum", "total", "combined"]):
        return "sum"
    if any(x in q for x in ["difference", "increase", "change", "reduction"]):
        return "diff"
    if "%" in q or "percent" in q:
        return "pct"
    
    return "unknown"


#  DATA LOADING
def load_predictions(path: str) -> Dict[str, Dict]:
    """Load predictions from JSON file, indexed by question_id."""
    with open(path) as f:
        preds = json.load(f)
    
    # Index by question_id
    indexed = {}
    for pred in preds:
        qid = pred.get("question_id")
        indexed[qid] = pred
    
    return indexed


def load_questions(csv_path: str) -> Dict[int, Dict]:
    """Load questions from CSV, indexed by row number."""
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    questions = {}
    for idx, row in df.iterrows():
        questions[idx] = {
            "question": row.get("question", ""),
            "gold_answer": safe_float(row.get("value")),
            "question_type_ext": row.get("question_type_ext", "unknown"),
            "pdf_name": row.get("pdf name", ""),
        }
    
    return questions


def build_dataset(rag_preds: Dict, agent_preds: Dict, questions: Dict) -> List[Dict]:
    """Match predictions to questions."""
    
    dataset = []
    
    for idx, question_data in questions.items():
        # Try to find matching predictions
        # RAG predictions and agent predictions should be at the same index
        
        # Find matching question_id - try both RAG and agent indices
        rag_match = None
        agent_match = None
        
        # Look through predictions for matches
        for qid in rag_preds:
            if str(idx) in qid or qid.endswith(f"-{idx}"):
                rag_match = rag_preds[qid]
                break
        
        for qid in agent_preds:
            if str(idx) in qid or qid.endswith(f"-{idx}"):
                agent_match = agent_preds[qid]
                break
        
        # If we found at least one, create entry
        if rag_match or agent_match:
            dataset.append({
                "question_id": idx,
                "question": question_data["question"],
                "gold": question_data["gold_answer"],
                
                "rag_answer": safe_float(rag_match.get("answer")) if rag_match else None,
                "rag_metadata": rag_match.get("metadata", {}) if rag_match else {},
                
                "agent_answer": safe_float(agent_match.get("answer")) if agent_match else None,
                "agent_metadata": agent_match.get("metadata", {}) if agent_match else {},
                
                "operation": detect_operation(question_data["question"]),
                "question_type": detect_question_type(question_data["question"]),
                "question_type_ext": question_data["question_type_ext"],
            })
    
    return dataset


#  CONTROLLED REASONING APPLICATION
def apply_controlled_reasoning(dataset: List[Dict]) -> List[Dict]:
    """Apply controlled reasoning wrapper to get fair comparison."""
    
    for i, item in enumerate(dataset):
        # Get retrieved chunks from either RAG or agent metadata
        chunks = []
        
        # Try RAG chunks first
        rag_meta = item.get("rag_metadata", {})
        if "retrieval_hits" in rag_meta:
            chunks = [hit.get("content_text", "") for hit in rag_meta.get("retrieval_hits", [])]
        
        # Fallback to agent chunks
        if not chunks:
            agent_meta = item.get("agent_metadata", {})
            if "retrieval_hits" in agent_meta:
                chunks = [hit.get("content_text", "") for hit in agent_meta.get("retrieval_hits", [])]
        
        # Apply controlled reasoning
        if chunks and item["gold"] is not None:
            try:
                result = controlled_reasoning(item["question"], chunks)
                item["controlled_answer"] = result.get("answer")
                item["controlled_operation"] = result.get("operation")
                item["controlled_error"] = None
            except Exception as e:
                item["controlled_answer"] = None
                item["controlled_operation"] = None
                item["controlled_error"] = str(e)
        else:
            item["controlled_answer"] = None
            item["controlled_operation"] = None
            item["controlled_error"] = "No chunks" if not chunks else "No gold answer"
    
    return dataset


#  EVALUATION
def evaluate(dataset: List[Dict]) -> Dict:
    """Compute metrics for all three systems using three threshold levels."""
    
    summary = {
        s: {"em": 0, "t20": 0, "t50": 0, "n": 0}
        for s in ["rag", "agentic", "controlled"]
    }
    summary["total"] = 0
    summary["total_with_gold"] = 0
    
    for item in dataset:
        gold = item["gold"]
        
        if gold is None:
            continue
        
        summary["total_with_gold"] += 1
        summary["total"] += 1
        
        for system, key in [("rag", "rag_answer"), ("agentic", "agent_answer"), ("controlled", "controlled_answer")]:
            pred = item[key]
            summary[system]["em"]  += exact_match(pred, gold)
            summary[system]["t20"] += tol_20(pred, gold)
            summary[system]["t50"] += tol_50(pred, gold)
            summary[system]["n"]   += 1
    
    # Normalize to percentages
    for system in ["rag", "agentic", "controlled"]:
        n = summary[system]["n"]
        if n > 0:
            summary[system]["em"]  = 100.0 * summary[system]["em"]  / n
            summary[system]["t20"] = 100.0 * summary[system]["t20"] / n
            summary[system]["t50"] = 100.0 * summary[system]["t50"] / n
    
    return summary


#  BREAKDOWN ANALYSIS
def breakdown(dataset: List[Dict], key: str) -> Dict:
    """Breakdown controlled reasoning performance by key (type, operation, etc)."""
    
    bucket = {}
    
    for item in dataset:
        if item["gold"] is None:
            continue
        
        k = item[key]
        
        if k not in bucket:
            bucket[k] = {
                "total": 0,
                "controlled_em": 0,
                "controlled_t20": 0,
                "controlled_t50": 0,
                "rag_em": 0,
                "agentic_em": 0,
            }
        
        bucket[k]["total"]          += 1
        bucket[k]["controlled_em"]  += exact_match(item["controlled_answer"], item["gold"])
        bucket[k]["controlled_t20"] += tol_20(item["controlled_answer"], item["gold"])
        bucket[k]["controlled_t50"] += tol_50(item["controlled_answer"], item["gold"])
        bucket[k]["rag_em"]         += exact_match(item["rag_answer"], item["gold"])
        bucket[k]["agentic_em"]     += exact_match(item["agent_answer"], item["gold"])
    
    # Compute percentages after all counts are final
    for k in bucket:
        n = bucket[k]["total"]
        bucket[k]["controlled_em_pct"]  = 100.0 * bucket[k]["controlled_em"]  / n
        bucket[k]["controlled_t20_pct"] = 100.0 * bucket[k]["controlled_t20"] / n
        bucket[k]["controlled_t50_pct"] = 100.0 * bucket[k]["controlled_t50"] / n
        bucket[k]["rag_em_pct"]         = 100.0 * bucket[k]["rag_em"]          / n
        bucket[k]["agentic_em_pct"]     = 100.0 * bucket[k]["agentic_em"]      / n
    
    return bucket


#  FAILURE COLLECTION
def collect_failures(dataset: List[Dict], n: int = 20) -> List[Dict]:
    """Collect worst failures from controlled reasoning."""
    
    failures = []
    
    for item in dataset:
        if item["gold"] is None:
            continue
        
        if exact_match(item["controlled_answer"], item["gold"]) == 0:
            failure = {
                "question_id": item["question_id"],
                "question": item["question"],
                "gold": item["gold"],
                "controlled_pred": item["controlled_answer"],
                "rag_pred": item["rag_answer"],
                "agentic_pred": item["agent_answer"],
                "operation": item["controlled_operation"],
                "error": item["controlled_error"],
            }
            failures.append(failure)
    
    return failures[:n]


#  MAIN
def main():
    """Run full benchmark."""
    
    base_dir = Path(__file__).parent.parent
    
    print("\n" + "=" * 90)
    print("FULL BENCHMARK: Traditional RAG vs Agentic vs Controlled Reasoning")
    print("=" * 90)
    
    # Load data
    print("\n📂 Loading data...")
    rag_preds = load_predictions(base_dir / "results/traditional_rag_predictions.json")
    agent_preds = load_predictions(base_dir / "results/agentic_multi_tool_predictions.json")
    questions = load_questions(base_dir / "data/benchmark/one-table/gri-qa_quant.csv")
    
    print(f"    {len(rag_preds)} RAG predictions")
    print(f"    {len(agent_preds)} Agentic predictions")
    print(f"    {len(questions)} questions")
    
    # Build dataset
    print("\n🔗 Matching predictions to questions...")
    dataset = build_dataset(rag_preds, agent_preds, questions)
    print(f"    {len(dataset)} samples with complete data")
    
    # Apply controlled reasoning
    print("\n🎯 Applying controlled reasoning wrapper...")
    dataset = apply_controlled_reasoning(dataset)
    print(f"    Done")
    
    # Save raw results
    output_path = base_dir / "results/full_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"    Saved to {output_path}")
    
    # Evaluation
    print("\n Evaluating performance...")
    summary = evaluate(dataset)
    
    print("\n" + "=" * 90)
    print("OVERALL RESULTS (Exact Match %)")
    print("=" * 90)
    print(f"{'System':<15} {'Exact (<1.0)':<15} {'Tol 20%':<12} {'Tol 50%':<12}")
    print("-" * 55)
    for system in ["rag", "agentic", "controlled"]:
        sys_name = system.upper()
        em  = summary[system]["em"]
        t20 = summary[system]["t20"]
        t50 = summary[system]["t50"]
        print(f"{sys_name:<15} {em:>6.2f}%{'':<8} {t20:>6.2f}%{'':<5} {t50:>6.2f}%")
    
    print(f"\nSamples evaluated: {summary['total_with_gold']}")
    
    # Breakdowns
    print("\n" + "=" * 90)
    print("BREAKDOWN BY QUESTION TYPE")
    print("=" * 90)
    
    type_breakdown = breakdown(dataset, "question_type")
    print(f"{'Type':<15} {'Count':<8} {'RAG EM%':>9} {'Agent EM%':>11} {'Ctrl EM%':>10} {'Ctrl Tol20%':>12} {'Ctrl Tol50%':>12}")
    print("-" * 80)
    for qtype in sorted(type_breakdown.keys()):
        data = type_breakdown[qtype]
        print(f"{qtype:<15} {data['total']:<8} {data['rag_em_pct']:>8.2f}% {data['agentic_em_pct']:>10.2f}% {data['controlled_em_pct']:>9.2f}% {data['controlled_t20_pct']:>11.2f}% {data['controlled_t50_pct']:>11.2f}%")
    
    print("\n" + "=" * 90)
    print("BREAKDOWN BY OPERATION")
    print("=" * 90)
    
    op_breakdown = breakdown(dataset, "operation")
    print(f"{'Operation':<15} {'Count':<8} {'RAG EM%':>9} {'Agent EM%':>11} {'Ctrl EM%':>10} {'Ctrl Tol20%':>12} {'Ctrl Tol50%':>12}")
    print("-" * 80)
    for op in sorted(op_breakdown.keys()):
        data = op_breakdown[op]
        print(f"{op:<15} {data['total']:<8} {data['rag_em_pct']:>8.2f}% {data['agentic_em_pct']:>10.2f}% {data['controlled_em_pct']:>9.2f}% {data['controlled_t20_pct']:>11.2f}% {data['controlled_t50_pct']:>11.2f}%")
    
    # Save failures
    print("\n" + "=" * 90)
    print("FAILURE ANALYSIS")
    print("=" * 90)
    
    failures = collect_failures(dataset, 20)
    failures_path = base_dir / "results/benchmark_failures.json"
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)
    
    print(f" Saved {len(failures)} failure cases to {failures_path}")
    
    if failures:
        print(f"\n🔴 Sample Failures:")
        for i, f in enumerate(failures[:3], 1):
            print(f"\n   Failure {i}:")
            print(f"     Q: {f['question'][:70]}...")
            print(f"     Gold: {f['gold']}")
            print(f"     Controlled: {f['controlled_pred']} (Op: {f['operation']})")
            if f['error']:
                print(f"     Error: {f['error']}")
    
    # Summary stats
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Metric':<25} {'RAG':>8} {'Agentic':>10} {'Controlled':>12}")
    print("-" * 58)
    print(f"{'Exact (< 1.0 abs)':<25} {summary['rag']['em']:>7.2f}% {summary['agentic']['em']:>9.2f}% {summary['controlled']['em']:>11.2f}%")
    print(f"{'Tolerance 20%':<25} {summary['rag']['t20']:>7.2f}% {summary['agentic']['t20']:>9.2f}% {summary['controlled']['t20']:>11.2f}%")
    print(f"{'Tolerance 50%':<25} {summary['rag']['t50']:>7.2f}% {summary['agentic']['t50']:>9.2f}% {summary['controlled']['t50']:>11.2f}%")
    print(f"\n📈 Controlled vs RAG (Tol 20%): {summary['controlled']['t20'] - summary['rag']['t20']:+.2f} pp")
    print(f"📈 Controlled vs Agentic (Tol 20%): {summary['controlled']['t20'] - summary['agentic']['t20']:+.2f} pp")
    print("\n")


if __name__ == "__main__":
    main()
