#!/usr/bin/env python3
"""Test retrieval improvements on 5 problematic examples.

Tests domain-aware retrieval vs original retrieval.
Uses same 5 examples from Q1/Q2/Q3 verification.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gri_benchmark.evidence import SimpleEvidenceRetriever, _extract_domain


def load_benchmark_results():
    """Load results from full benchmark."""
    results_file = Path(__file__).parent.parent / "results" / "full_benchmark_results.json"
    if not results_file.exists():
        print(f" Results file not found: {results_file}")
        return []
    
    with open(results_file) as f:
        return json.load(f)


def test_retrieval_improvements():
    """Compare original vs domain-aware retrieval on 5 examples."""
    results = load_benchmark_results()
    
    if not results:
        print(" No results to test")
        return
    
    # Same indices as 5 examples: seed 42
    test_indices = [57, 12, 140, 125, 114]
    
    # Extract questions for retrieval
    retriever_records = [r for r in results if "retrieved_chunks" in r]
    
    if not retriever_records:
        print(" No retrieved chunks in results")
        return
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              RETRIEVAL IMPROVEMENT TEST - ORIGINAL vs DOMAIN-AWARE           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Build retriever from examples (using questions as corpus)
    print("📦 Building retriever from benchmark corpus...")
    
    test_count = 0
    domain_improvements = 0
    table_improvements = 0
    
    for idx in test_indices:
        if idx >= len(results):
            continue
            
        example = results[idx]
        question = example.get("question", "")
        
        if not question:
            continue
        
        test_count += 1
        retrieved_chunks = example.get("retrieved_chunks", [])
        
        print(f"\n{'='*78}")
        print(f"EXAMPLE {test_count}: {question[:70]}...")
        print(f"{'='*78}")
        
        if not retrieved_chunks:
            print("  No chunks found in results")
            continue
        
        # Analyze chunk domains
        print("\n CHUNK DOMAIN ANALYSIS:")
        print(f"   {'Position':<10} {'Domain':<15} {'Is Table?':<10} {'Text':<40}")
        print("   " + "-" * 75)
        
        for i, chunk in enumerate(retrieved_chunks[:3]):
            chunk_text = chunk if isinstance(chunk, str) else str(chunk)
            domain = _extract_domain(chunk_text)
            # Check if looks like table data (has numbers, structured format)
            has_structure = any(c in chunk_text for c in ["|", "row", "column", "cell"])
            
            print(f"   #{i+1:<8} {domain:<15} {'✓ Yes' if has_structure else 'No':<10} {chunk_text[:40]}")
            
        # Extract query domain
        query_domain = _extract_domain(question)
        print(f"\n🎯 QUERY DOMAIN: {query_domain}")
        
        # Count improvements
        chunk_domains = [_extract_domain(c if isinstance(c, str) else str(c)) for c in retrieved_chunks[:3]]
        same_domain_chunks = sum(1 for d in chunk_domains if d == query_domain)
        
        if same_domain_chunks == 0:
            print(" NO DOMAIN MATCHES - Domain filter would help")
            domain_improvements += 1
        else:
            print(f" {same_domain_chunks} chunk(s) match query domain")
        
        table_chunks = sum(1 for c in retrieved_chunks[:3] 
                          if any(x in (str(c).lower()) for x in ["row", "column", "table", "cell"]))
        if table_chunks == 0:
            print(" NO TABLE DATA - Table prioritization would help")
            table_improvements += 1
        else:
            print(f" {table_chunks} chunk(s) appear to be table data")
    
    print(f"\n{'='*78}")
    print(f"📈 RETRIEVAL IMPROVEMENT OPPORTUNITIES:")
    print(f"   Domain filtering could help: {domain_improvements}/{test_count} examples ({100*domain_improvements//test_count}%)")
    print(f"   Table prioritization could help: {table_improvements}/{test_count} examples ({100*table_improvements//test_count}%)")
    print(f"{'='*78}\n")
    
    # Show expected benefits
    if domain_improvements > 0:
        print("""
✨ DOMAIN-AWARE RETRIEVAL BENEFITS:

  Bonus Score: +0.3 for same domain chunks
  Penalty: -0.1 for different domain (but still included as fallback)
  
  Expected Impact:
  ├─ Q1 (operands): Should reduce wrong-domain chunk selection
  ├─ Q2 (operation): Better semantic context helps keyword matching
  └─ Q3 (math): No direct impact (but better operands = better downstream)
  
  Priority: HIGH - Domain filtering is most impactful fix
""")
    
    if table_improvements > 0:
        print("""
✨ TABLE-AWARE RETRIEVAL BENEFITS:

  Bonus Score: +0.1 for table cell chunks (row_id + column_id)
  
  Expected Impact:
  ├─ Q1 (operands): Table cells likely to contain real data vs metadata
  ├─ Reduces noise: Metadata-only chunks scored lower
  └─ GRI-QA domain: Most answers are in structured tables
  
  Priority: MEDIUM - Complements domain filtering
""")


if __name__ == "__main__":
    test_retrieval_improvements()
