#!/usr/bin/env python3
"""STEP 1: Isolated retrieval comparison (CORRECTED).

Uses ACTUAL retrieved chunks from results, re-scores with old vs new methods.
This shows if domain-aware scoring would have ranked chunks better.

Approach:
1. For each question, get its actual retrieval_hits
2. Re-score each hit with: OLD scoring (lexical only) vs NEW scoring (domain-aware)
3. Compare ranking: would top-1 have changed? Would it be better?
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gri_benchmark.evidence import _extract_domain


def load_benchmark_results() -> list[dict]:
    """Load benchmark results with retrieval metadata."""
    results_file = Path(__file__).parent.parent / "results" / "full_benchmark_results.json"
    if not results_file.exists():
        print(f" Results file not found: {results_file}")
        return []
    
    with open(results_file) as f:
        return json.load(f)


def calculate_old_score(hit: dict) -> float:
    """Calculate score using OLD method (lexical only)."""
    breakdown = hit.get("score_breakdown", {})
    lexical = breakdown.get("lexical", 0.0)
    intent_match = breakdown.get("intent_match", 0.0)
    # Old formula: lexical + 0.15 * intent_match
    return lexical + (0.15 * intent_match)


def calculate_new_score(hit: dict, query_domain: str) -> float:
    """Calculate score using NEW method (with domain bonus)."""
    breakdown = hit.get("score_breakdown", {})
    lexical = breakdown.get("lexical", 0.0)
    intent_match = breakdown.get("intent_match", 0.0)
    
    # Domain bonus
    content_text = hit.get("content_text", "")
    chunk_domain = _extract_domain(content_text)
    
    if chunk_domain == "other" or query_domain == "other":
        domain_bonus = 0.0
    elif chunk_domain == query_domain:
        domain_bonus = 0.3
    else:
        domain_bonus = -0.1
    
    # Table bonus
    has_row = hit.get("row_id") is not None
    has_col = hit.get("column_id") is not None
    table_bonus = 0.1 if (has_row and has_col) else 0.0
    
    # New formula
    return lexical + (0.15 * intent_match) + domain_bonus + table_bonus


def compare_retrievals_on_actual_corpus(num_samples: int = 15):
    """Compare old vs new scoring on actual retrieved chunks."""
    results = load_benchmark_results()
    
    if not results:
        print(" No results")
        return
    
    # Filter to examples with retrieval hits
    examples_with_hits = [
        (idx, ex) for idx, ex in enumerate(results)
        if ex.get("rag_metadata", {}).get("retrieval_hits")
    ]
    
    if not examples_with_hits:
        print(" No examples with retrieval hits")
        return
    
    # Take first N
    examples_to_test = examples_with_hits[:num_samples]
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    STEP 1: ISOLATED RETRIEVAL COMPARISON                     ║
║           Re-scoring ACTUAL chunks: Old (Lexical) vs New (Domain)            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    old_top_domain_match = 0
    new_top_domain_match = 0
    ranking_improvements = 0
    ranking_regressions = 0
    
    for ex_num, (idx, example) in enumerate(examples_to_test, 1):
        question = example.get("question", "")
        gold = example.get("gold", "?")
        query_domain = _extract_domain(question)
        
        rag_metadata = example.get("rag_metadata", {})
        retrieval_hits = rag_metadata.get("retrieval_hits", [])
        
        if not retrieval_hits:
            continue
        
        # Re-score all hits with both methods
        scored_hits_old = []
        scored_hits_new = []
        
        for hit in retrieval_hits:
            old_score = calculate_old_score(hit)
            new_score = calculate_new_score(hit, query_domain)
            
            scored_hits_old.append((hit, old_score))
            scored_hits_new.append((hit, new_score))
        
        # Sort by score
        scored_hits_old.sort(key=lambda x: x[1], reverse=True)
        scored_hits_new.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-1
        old_top_hit, old_top_score = scored_hits_old[0]
        new_top_hit, new_top_score = scored_hits_new[0]
        
        # Analyze
        old_content = old_top_hit.get("content_text", "")
        new_content = new_top_hit.get("content_text", "")
        
        old_domain = _extract_domain(old_content)
        new_domain = _extract_domain(new_content)
        
        old_match = (old_domain == query_domain)
        new_match = (new_domain == query_domain)
        
        if old_match:
            old_top_domain_match += 1
        if new_match:
            new_top_domain_match += 1
        
        # Track ranking changes
        if new_match and not old_match:
            ranking_improvements += 1
        elif old_match and not new_match:
            ranking_regressions += 1
        
        # Display
        print(f"\n{'='*78}")
        print(f"EXAMPLE {ex_num} (idx {idx})")
        print(f"{'='*78}")
        print(f"Q: {question[:70]}...")
        print(f"Gold: {gold:<15} Query Domain: {query_domain.upper()}")
        
        print(f"\n🔴 OLD RETRIEVAL (Lexical-Only) - Top-1 Result:")
        print(f"   Score: {old_top_score:.3f}")
        print(f"   Domain: {old_domain:<15} {' MATCH' if old_match else ' MISMATCH'}")
        print(f"   Content: {old_content[:65]}...")
        print(f"   Ranking: #{[h[0]['record_id'] for h in scored_hits_old].index(old_top_hit['record_id']) + 1}")
        
        print(f"\n🟢 NEW RETRIEVAL (Domain-Aware) - Top-1 Result:")
        print(f"   Score: {new_top_score:.3f}")
        print(f"   Domain: {new_domain:<15} {' MATCH' if new_match else ' MISMATCH'}")
        print(f"   Content: {new_content[:65]}...")
        print(f"   Ranking: #{[h[0]['record_id'] for h in scored_hits_new].index(new_top_hit['record_id']) + 1}")
        
        # Improvement indicator
        if new_top_hit == old_top_hit:
            if new_match:
                print(f"\n    SAME TOP, BOTH CORRECT (no change needed)")
            else:
                print(f"\n   ⚪ SAME TOP, BOTH WRONG (no change)")
        elif new_match and not old_match:
            print(f"\n   🎯 IMPROVEMENT: Domain-aware picked better chunk!")
        elif old_match and not new_match:
            print(f"\n     REGRESSION: Domain-aware picked worse chunk")
        else:
            print(f"\n   📊 CHANGE: Different chunk selected, both {'correct' if new_match else 'wrong'}")
    
    # Summary
    print(f"\n{'='*78}")
    print(f"📊 STEP 1 SUMMARY - ACTUAL CORPUS RE-SCORING")
    print(f"{'='*78}")
    
    num_tested = len(examples_to_test)
    
    print(f"\nTop-1 Domain Match Rate:")
    print(f"   OLD: {old_top_domain_match}/{num_tested} ({100*old_top_domain_match//num_tested}%)")
    print(f"   NEW: {new_top_domain_match}/{num_tested} ({100*new_top_domain_match//num_tested}%)")
    print(f"   Δ:   {new_top_domain_match - old_top_domain_match:+d} ({100*(new_top_domain_match - old_top_domain_match)//num_tested:+d}pp)")
    
    print(f"\nRanking Changes:")
    print(f"   Improvements: {ranking_improvements}")
    print(f"   Regressions: {ranking_regressions}")
    print(f"   Net: {ranking_improvements - ranking_regressions:+d}")
    
    # Decision logic
    print(f"\n{'='*78}")
    improvement_pct = 100 * (new_top_domain_match - old_top_domain_match) // max(num_tested, 1)
    
    if improvement_pct >= 15:
        print(f" STRONG IMPROVEMENT: +{improvement_pct}pp domain match")
        print(f"   Decision: PROCEED TO STEP 2 ")
        print(f"   Rationale: Significant improvement in retrieval quality")
    elif improvement_pct >= 5:
        print(f"  MODERATE IMPROVEMENT: +{improvement_pct}pp domain match")
        print(f"   Decision: PROCEED TO STEP 2  (monitor closely)")
        print(f"   Rationale: Measurable improvement, worth testing")
    elif improvement_pct == 0 and ranking_improvements > 0:
        print(f"⚡ TARGETED IMPROVEMENTS: {ranking_improvements} cases improved")
        print(f"   Decision: PROCEED TO STEP 2  (low risk)")
        print(f"   Rationale: Even if avg %, targeted improvements exist")
    else:
        print(f" NO IMPROVEMENT: {improvement_pct:+d}pp (possibly negative)")
        print(f"   Decision: STOP - Not worth pursuing 🛑")
        print(f"   Rationale: Domain-aware retrieval doesn't improve ranking")
    
    print(f"{'='*78}\n")


if __name__ == "__main__":
    compare_retrievals_on_actual_corpus(num_samples=15)
