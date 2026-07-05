#!/usr/bin/env python3
"""Analyze why agentic fails where RAG succeeds."""
import json
from pathlib import Path
import pandas as pd

# Load predictions
with open('results/full/agentic_multi_tool_predictions.json') as f:
    agentic = json.load(f)

with open('results/full/traditional_rag_predictions.json') as f:
    rag = json.load(f)

# Build gold map
gold_map = {}
for csv in Path('data/benchmark').glob('*/*.csv'):
    df = pd.read_csv(csv)
    for _, row in df.iterrows():
        qid = row.get('question_id', '')
        if qid and pd.notna(row.get('answer')):
            try:
                gold_map[qid] = float(row['answer'])
            except:
                pass

# Maps
rag_map = {p['question_id']: p for p in rag}
agentic_map = {p['question_id']: p for p in agentic}

# Find failures
agentic_failed_rag_ok = []
for qid in agentic_map:
    if qid not in gold_map or qid not in rag_map:
        continue
    
    gold = gold_map[qid]
    ag_ans = agentic_map[qid].get('answer')
    rag_ans = rag_map[qid].get('answer')
    q_text = agentic_map[qid].get('question', '')
    
    try:
        ag_float = float(ag_ans)
        rag_float = float(rag_ans)
    except:
        continue
    
    ag_correct = abs(ag_float - gold) < 1.0
    rag_correct = abs(rag_float - gold) < 1.0
    
    if not ag_correct and rag_correct:
        agentic_failed_rag_ok.append({
            'qid': qid,
            'question': q_text[:100],
            'gold': gold,
            'ag_ans': ag_float,
            'rag_ans': rag_float,
            'ag_err': abs(ag_float - gold),
            'rag_err': abs(rag_float - gold),
        })

print(f"\n{'='*80}")
print(f"AGENTIC FAILURES (where RAG succeeded)")
print(f"{'='*80}")
print(f"Total: {len(agentic_failed_rag_ok)} cases out of {len(gold_map)} questions\n")

# Sort by error magnitude
sorted_failures = sorted(agentic_failed_rag_ok, key=lambda x: x['ag_err'], reverse=True)

print(f"TOP 15 WORST AGENTIC FAILURES:")
print(f"{'-'*80}\n")

for i, case in enumerate(sorted_failures[:15], 1):
    print(f"{i}. Question: {case['question']}")
    print(f"   Gold Answer: {case['gold']}")
    print(f"   RAG Answer:     {case['rag_ans']:.1f} (error: {case['rag_err']:.1f})")
    print(f"   Agentic Answer: {case['ag_ans']:.1f} (error: {case['ag_err']:.1f})")
    print(f"   Delta: Agentic is {case['ag_err'] - case['rag_err']:.1f} units worse\n")

# Categorize by error magnitude
huge_errors = [c for c in sorted_failures if c['ag_err'] > 1000]
large_errors = [c for c in sorted_failures if 100 < c['ag_err'] <= 1000]
medium_errors = [c for c in sorted_failures if 10 < c['ag_err'] <= 100]
small_errors = [c for c in sorted_failures if c['ag_err'] <= 10]

print(f"\nERROR DISTRIBUTION:")
print(f"  Huge (>1000):    {len(huge_errors):3d} ({len(huge_errors)/len(sorted_failures)*100:5.1f}%)")
print(f"  Large (100-1K):  {len(large_errors):3d} ({len(large_errors)/len(sorted_failures)*100:5.1f}%)")
print(f"  Medium (10-100): {len(medium_errors):3d} ({len(medium_errors)/len(sorted_failures)*100:5.1f}%)")
print(f"  Small (<10):     {len(small_errors):3d} ({len(small_errors)/len(sorted_failures)*100:5.1f}%)")

# Analyze patterns
print(f"\n\nPATTERN ANALYSIS:")
print(f"{'-'*80}")

# Look for operation type patterns
operation_words = {
    'sum': 'sum|total|combined|aggregate',
    'average': 'average|mean|median|per',
    'percentage': 'percent|%|ratio|proportion',
    'difference': 'increase|decrease|reduction|change|difference',
    'max/min': 'maximum|minimum|highest|lowest|largest|smallest|top|leading',
}

for op_type, keywords in operation_words.items():
    import re
    count = sum(1 for c in sorted_failures if re.search(keywords, c['question'].lower()))
    if count > 0:
        print(f"  {op_type:20s}: {count:3d} failures")
