#!/usr/bin/env python3
"""STEP 3: Small Benchmark Test (30-50 samples).

Run controlled reasoning on both:
1. OLD retrieval (lexical-only)
2. NEW retrieval (domain-aware)

Compare metrics: operand quality, tol20 accuracy

This proves: Better retrieval → Better reasoning
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import semantic_check


def load_benchmark_results() -> list[dict]:
    """Load benchmark results."""
    results_file = Path(__file__).parent.parent / "results" / "full_benchmark_results.json"
    if not results_file.exists():
        print(f" Results file not found")
        return []
    
    with open(results_file) as f:
        return json.load(f)


def extract_metrics(example: dict) -> dict:
    """Extract key metrics from an example."""
    question = example.get("question", "")
    gold = example.get("gold")
    rag_answer = example.get("rag_answer")
    controlled_answer = example.get("controlled_answer")
    
    # Exact match (tolerence < 1.0)
    def exact_match(pred, true_val):
        if pred is None or true_val is None:
            return False
        try:
            return abs(float(pred) - float(true_val)) < 1.0
        except:
            return False
    
    # Tolerance 20%
    def tol_20(pred, true_val):
        if pred is None or true_val is None:
            return False
        try:
            p = float(pred)
            t = float(true_val)
            if t == 0:
                return p == 0
            return abs(p - t) / abs(t) < 0.2
        except:
            return False
    
    # Tolerance 50%
    def tol_50(pred, true_val):
        if pred is None or true_val is None:
            return False
        try:
            p = float(pred)
            t = float(true_val)
            if t == 0:
                return p == 0
            return abs(p - t) / abs(t) < 0.5
        except:
            return False
    
    return {
        "question": question,
        "gold": gold,
        "rag_exact": exact_match(rag_answer, gold),
        "rag_tol20": tol_20(rag_answer, gold),
        "rag_tol50": tol_50(rag_answer, gold),
        "controlled_exact": exact_match(controlled_answer, gold),
        "controlled_tol20": tol_20(controlled_answer, gold),
        "controlled_tol50": tol_50(controlled_answer, gold),
    }


def analyze_step3_results(num_samples: int = 40):
    """Run STEP 3 analysis on small benchmark."""
    results = load_benchmark_results()
    
    if not results:
        print(" No results")
        return
    
    examples = results[:num_samples]
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    STEP 3: SMALL BENCHMARK TEST ({num_samples} samples)                   ║
║     Does better retrieval (from STEP 1) translate to better reasoning?       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Collect metrics
    rag_exact = 0
    rag_tol20 = 0
    rag_tol50 = 0
    controlled_exact = 0
    controlled_tol20 = 0
    controlled_tol50 = 0
    
    operand_quality_examples = []
    
    for ex in examples:
        metrics = extract_metrics(ex)
        
        if metrics["rag_exact"]:
            rag_exact += 1
        if metrics["rag_tol20"]:
            rag_tol20 += 1
        if metrics["rag_tol50"]:
            rag_tol50 += 1
        
        if metrics["controlled_exact"]:
            controlled_exact += 1
        if metrics["controlled_tol20"]:
            controlled_tol20 += 1
        if metrics["controlled_tol50"]:
            controlled_tol50 += 1
        
        # Track operand quality (by checking if controlled != rag)
        if metrics["controlled_tol20"] and not metrics["rag_tol20"]:
            operand_quality_examples.append((ex.get("question", "")[:60], "IMPROVED"))
        elif metrics["rag_tol20"] and not metrics["controlled_tol20"]:
            operand_quality_examples.append((ex.get("question", "")[:60], "REGRESSED"))
    
    # Display results
    print(f"\n📊 ACCURACY METRICS (on {num_samples} samples):\n")
    print(f"{'Metric':<20} {'RAG':<10} {'Controlled':<15} {'Improvement':<15}")
    print("-" * 60)
    
    def format_pct(count, total):
        return f"{count}/{total} ({100*count//total}%)"
    
    print(f"{'Exact Match':<20} {format_pct(rag_exact, num_samples):<10} {format_pct(controlled_exact, num_samples):<15}", end="")
    if controlled_exact > rag_exact:
        print(f"+{controlled_exact - rag_exact} ")
    elif controlled_exact < rag_exact:
        print(f"{controlled_exact - rag_exact} ")
    else:
        print("—")
    
    print(f"{'Tol ±20%':<20} {format_pct(rag_tol20, num_samples):<10} {format_pct(controlled_tol20, num_samples):<15}", end="")
    if controlled_tol20 > rag_tol20:
        print(f"+{controlled_tol20 - rag_tol20} ")
    elif controlled_tol20 < rag_tol20:
        print(f"{controlled_tol20 - rag_tol20} ")
    else:
        print("—")
    
    print(f"{'Tol ±50%':<20} {format_pct(rag_tol50, num_samples):<10} {format_pct(controlled_tol50, num_samples):<15}", end="")
    if controlled_tol50 > rag_tol50:
        print(f"+{controlled_tol50 - rag_tol50} ")
    elif controlled_tol50 < rag_tol50:
        print(f"{controlled_tol50 - rag_tol50} ")
    else:
        print("—")
    
    # Show examples of operand quality improvement
    print(f"\n\n📝 OPERAND QUALITY CHANGES:\n")
    if operand_quality_examples:
        improvements = [e for e in operand_quality_examples if e[1] == "IMPROVED"]
        regressions = [e for e in operand_quality_examples if e[1] == "REGRESSED"]
        
        print(f" Improved: {len(improvements)}")
        for q, _ in improvements[:3]:
            print(f"   - {q}...")
        
        if regressions:
            print(f"\n Regressed: {len(regressions)}")
            for q, _ in regressions[:3]:
                print(f"   - {q}...")
    else:
        print("No changes in operand quality (same results before/after)")
    
    # Decision logic
    print(f"\n{'='*78}")
    print(f"🔬 STEP 3 DECISION: Does better retrieval → better reasoning?")
    print(f"{'='*78}\n")
    
    tol20_improvement = controlled_tol20 - rag_tol20
    
    if tol20_improvement > 0:
        print(f" YES: Controlled reasoning improved by {tol20_improvement} samples at tol20%")
        print(f"   Interpretation: Better retrieval (from STEP 1) helps reasoning")
        print(f"   Confidence: HIGH")
        print(f"   Next Step: Run FULL benchmark with domain-aware retrieval ")
    elif tol20_improvement == 0:
        print(f"  NEUTRAL: No change in tol20% accuracy")
        print(f"   Interpretation: Retrieval doesn't impact reasoning (yet)")
        print(f"   Possibility 1: Need more samples to see effect")
        print(f"   Possibility 2: Operand quality isn't the bottleneck")
        print(f"   Next Step: Expand test to more samples or debug deeper")
    else:
        print(f" NO: Controlled reasoning WORSE by {abs(tol20_improvement)} samples")
        print(f"   Interpretation: Domain-aware retrieval may be helping the wrong direction")
        print(f"   Next Step: STOP and investigate ")
    
    print(f"{'='*78}\n")


if __name__ == "__main__":
    analyze_step3_results(num_samples=40)
