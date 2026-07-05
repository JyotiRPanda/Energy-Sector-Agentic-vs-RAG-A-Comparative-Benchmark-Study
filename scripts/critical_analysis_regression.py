#!/usr/bin/env python3
"""CRITICAL ANALYSIS: Domain-Aware vs Baseline Comparison (266 samples).

FINDING: Full benchmark shows unexpected REGRESSION with domain-aware retrieval.
This contradicts STEP 3 small benchmark validation results.
"""
import json

# Baseline results (WITHOUT domain-aware flag)
baseline = {
    "traditional_rag": {
        "exact_match": 0.34407935523868566,
        "citation_precision": 0.5716057036577805,
        "faithfulness": 0.7858028518288903,
        "error.incorrect_quantitative_operation": 0.4922504649721017,
        "error.miscitation": 0.4283942963422195,
        "latency_retrieval_ms": 1.346021477681342,
    },
    "agentic_multi_tool": {
        "exact_match": 0.3180409175449473,
        "citation_precision": 0.520252118206241,
        "faithfulness": 0.708152510849349,
        "error.incorrect_quantitative_operation": 0.5216986980781153,
        "error.miscitation": 0.4962802231866088,
        "latency_retrieval_ms": 1.4076659119652821,
    }
}

# Domain-aware results (WITH domain-aware flag)
domain_aware = {
    "traditional_rag": {
        "exact_match": 0.3267203967761934,
        "citation_precision": 0.5316181029138252,
        "faithfulness": 0.7658090514569126,
        "error.incorrect_quantitative_operation": 0.5,
        "error.miscitation": 0.46838189708617484,
        "latency_retrieval_ms": 1.9121041041537503,
    },
    "agentic_multi_tool": {
        "exact_match": 0.3013019218846869,
        "citation_precision": 0.4857408555486671,
        "faithfulness": 0.6909485430874147,
        "error.incorrect_quantitative_operation": 0.527588344699318,
        "error.miscitation": 0.5319280843149411,
        "latency_retrieval_ms": 1.9857435985741287,
    }
}


def compare():
    """Compare baseline vs domain-aware."""
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      CRITICAL FINDING: REGRESSION DETECTED               ║
║           Domain-Aware Retrieval WORSE than Baseline (Full Benchmark)        ║
╚══════════════════════════════════════════════════════════════════════════════╝

🔍 COMPARISON: Baseline (lexical-only) vs Domain-Aware (full 266 samples)

""")
    
    for pipeline in ["traditional_rag", "agentic_multi_tool"]:
        print(f"\n{'='*78}")
        print(f"PIPELINE: {pipeline.upper()}")
        print(f"{'='*78}\n")
        
        print(f"{'Metric':<35} {'Baseline':<15} {'Domain-Aware':<15} {'Δ':<15}")
        print("-" * 80)
        
        baseline_metrics = baseline[pipeline]
        domain_metrics = domain_aware[pipeline]
        
        for metric_name in sorted(baseline_metrics.keys()):
            base_val = baseline_metrics[metric_name]
            domain_val = domain_metrics[metric_name]
            delta = domain_val - base_val
            
            # Format values
            if metric_name.endswith("_ms"):
                base_str = f"{base_val:.2f}ms"
                domain_str = f"{domain_val:.2f}ms"
                delta_str = f"{delta:+.2f}ms"
            else:
                base_pct = base_val * 100
                domain_pct = domain_val * 100
                delta_pct = delta * 100
                base_str = f"{base_pct:.1f}%"
                domain_str = f"{domain_pct:.1f}%"
                delta_str = f"{delta_pct:+.1f}pp"
            
            # Color indicator
            if delta < -0.001:
                indicator = " WORSE"
            elif delta > 0.001:
                indicator = " BETTER"
            else:
                indicator = "—"
            
            print(f"{metric_name:<35} {base_str:<15} {domain_str:<15} {delta_str:<15} {indicator}")
    
    print(f"\n{'='*78}")
    print(f" SUMMARY STATISTICS:")
    print(f"{'='*78}\n")
    
    # Calculate overall changes
    rag_changes = []
    agent_changes = []
    
    for metric in ["exact_match", "citation_precision", "faithfulness"]:
        rag_delta = domain_aware["traditional_rag"][metric] - baseline["traditional_rag"][metric]
        agent_delta = domain_aware["agentic_multi_tool"][metric] - baseline["agentic_multi_tool"][metric]
        rag_changes.append(rag_delta)
        agent_changes.append(agent_delta)
    
    rag_avg_change = sum(rag_changes) / len(rag_changes)
    agent_avg_change = sum(agent_changes) / len(agent_changes)
    
    print(f"Traditional RAG:")
    print(f"  Exact Match:       {(domain_aware['traditional_rag']['exact_match'] - baseline['traditional_rag']['exact_match'])*100:+.1f}pp")
    print(f"  Citation Precision: {(domain_aware['traditional_rag']['citation_precision'] - baseline['traditional_rag']['citation_precision'])*100:+.1f}pp")
    print(f"  Faithfulness:      {(domain_aware['traditional_rag']['faithfulness'] - baseline['traditional_rag']['faithfulness'])*100:+.1f}pp")
    print(f"  Average Change:    {rag_avg_change*100:+.1f}pp  REGRESSION")
    
    print(f"\nAgentic Multi-Tool:")
    print(f"  Exact Match:       {(domain_aware['agentic_multi_tool']['exact_match'] - baseline['agentic_multi_tool']['exact_match'])*100:+.1f}pp")
    print(f"  Citation Precision: {(domain_aware['agentic_multi_tool']['citation_precision'] - baseline['agentic_multi_tool']['citation_precision'])*100:+.1f}pp")
    print(f"  Faithfulness:      {(domain_aware['agentic_multi_tool']['faithfulness'] - baseline['agentic_multi_tool']['faithfulness'])*100:+.1f}pp")
    print(f"  Average Change:    {agent_avg_change*100:+.1f}pp  REGRESSION")
    
    print(f"""
 CRITICAL ISSUE DISCOVERED
═════════════════════════════════════════════════════════════════════════════════

Domain-aware retrieval resulted in WORSE performance across the board:

  Traditional RAG:
    • Exact Match:       34.4% → 32.7% (-1.7pp) 
    • Citation Precision: 57.2% → 53.2% (-4.0pp) 
    • Faithfulness:      78.6% → 76.6% (-2.0pp) 

  Agentic Multi-Tool:
    • Exact Match:       31.8% → 30.1% (-1.7pp) 
    • Citation Precision: 52.0% → 48.6% (-3.4pp) 
    • Faithfulness:      70.8% → 69.1% (-1.7pp) 

Error Rates INCREASED:
    • Traditional RAG miscitation: 42.8% → 46.8% (+4.0pp)
    • Agentic miscitation: 49.6% → 53.2% (+3.6pp)

Performance Latency:
    • Retrieval added ~0.6ms overhead with NO accuracy benefit

🔍 ROOT CAUSE ANALYSIS
═════════════════════════════════════════════════════════════════════════════════

Why did validation show improvement but full benchmark shows regression?

1. STEP 3 Validation vs Full Benchmark Difference:
   • STEP 3 used CONTROLLED reasoning (semantic_check module)
   • Full benchmark uses RAG and agentic pipelines (different reasoning)
   • Controlled reasoning may have different characteristics than end-to-end pipelines

2. Domain Classification May Be Too Aggressive:
   • Soft penalty (-0.1) might be excessive
   • Many relevant chunks get penalized despite being useful
   • Example: Energy question might retrieve emissions data if cross-domain relationship exists

3. Small Benchmark Effect:
   • 40-sample test might have been unrepresentative
   • Full 266 samples reveals the broader problem
   • Domain distribution might differ in unseen samples

4. Wrong-Domain Chunks May Still Be Useful:
   • Some questions need cross-domain reasoning
   • Waste and emissions questions might use same keywords
   • Forcing domain alignment could harm valid inferences

🛑 RECOMMENDATION: REVERT DOMAIN-AWARE RETRIEVAL
═════════════════════════════════════════════════════════════════════════════════

The domain-aware retrieval experiment has FAILED on full benchmark.

Action Items:
  1.  REVERT --use-domain-aware-retrieval flag from production
  2. 📋 Document failure in methodology
  3. 🔬 Investigate root causes before re-attempting
  4. 💡 Consider alternative approaches:
     - Softer penalties (e.g., -0.05 instead of -0.1)
     - Dynamic domain weighting based on question type
     - Ensemble approach: blend lexical and domain-aware scores
     - Multi-stage retrieval: try domain-aware, fall back to lexical if empty

Next Meeting Recommendation:
  • Analyze why STEP 3 showed improvements but STEP 5 shows regression
  • Hypothesis: Validation methodology doesn't transfer to full pipeline
  • Need to redesign validation to use actual RAG/agentic pipelines, not controlled reasoning

""")


if __name__ == "__main__":
    compare()
