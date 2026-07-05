# Operation-Specific Analysis: Verification of Fixes

**Report Date:** June 27, 2024  
**Benchmark:** single_table_quantitative (266 samples)  
**Pipeline:** agentic_multi_tool  

## Executive Summary

✗ **NO IMPROVEMENT** in critical quantitative operations:
- Sum: 0/57 (0.0%) 
- Average: 0/60 (0.0%)
- Reduction Difference: 0/70 (0.0%)

Overall accuracy: **26.2%** (mostly from extractive operations at 36%)

## Root Cause Analysis

### Current Status: RETRIEVAL FAILURE (Not Calculation)

The calculation engine is **working correctly**. The problem is that it receives the **wrong cells** from retrieval.

#### Example Case: SUM Question
```
Question: "What is the sum of the total waste in tons for the years 2022 and 2023?"
Gold Answer: 9420.0 (this is already the correct sum)

Current Behavior:
- Retrieves: rows [3, 9, 33] with values [16.3, 17382.5, 9420.0]
- Calculates: 16.3 + 17382.5 + 9420.0 = 26818.8
- Prediction: 26818.8 ❌ WRONG

Expected Behavior:
- Should retrieve: only row 33, col 4 with value 9420.0
- Should calculate: sum of just [9420.0] = 9420.0
- Prediction: 9420.0 ✅ CORRECT
```

### Why is Retrieval Failing?

1. **Corpus Structure Problem:**
   - Corpus has MULTIPLE records for the same (table_id, row_id, column_id)
   - Different records represent different questions' answers
   - Example: (table=0, row=33, col=4) has records for:
     - 4350 (for a different question about 2023)
     - 9420.0 (for the sum question)
     - Other values for other questions

2. **Row Enrichment Not Filtering:**
   - `search_with_row_joining()` combines ALL cells from a row
   - This includes cells from different question contexts
   - Results in heterogeneous cell maps with conflicting values

3. **Retrieval Returning Wrong Rows:**
   - Top-k results include rows 3, 9, and 33
   - Should only include row 33
   - This suggests retrieval ranking is poor for quantitative queries

## What Worked / What Didn't

### ✗ Changes That Didn't Improve Accuracy

1. **Removed `is_table_data` guards** (evidence.py line 897)
   - Enabled row_index population
   - Did NOT improve retrieval quality

2. **Changed to `search_with_row_joining()`** (tools.py)
   - Added row enrichment to retrieval
   - Returns incorrect multi-row results

3. **Modified value extraction to group by year** (calculation_engine.py)
   - Works correctly for values it receives
   - But receives wrong values from retrieval

4. **Fixed table_aware_grounding guards**
   - Only relevant for table_aware pipeline (not agentic_multi_tool)
   - Main pipeline still using basic calculation engine

## Key Insight: The Real Problem

**The agentic_multi_tool pipeline doesn't use table-aware grounding!**

- `table_aware_grounding.py` contains sophisticated per-row operand selection
- `agentic_multi_tool_pipeline.py` uses simpler `calculation_engine.py`
- Calculation engine has NO awareness of row/column structure
- It just extracts ALL numeric values and sums them

## Next Steps

### Option A: Use Table-Aware Pipeline (Recommended)
- Switch to `agentic_table_aware` pipeline in benchmark config
- This pipeline uses `table_aware_grounding.py` which has proper row/column logic
- Would activate all the table identity fixes

### Option B: Fix Retrieval Ranking
- Make lexical scoring better for quantitative queries
- Add domain weighting to prefer correct metric columns
- Filter corpus by question context instead of just metrics

### Option C: Implement Question-Context Filtering
- Instead of returning all cells from a row, filter by question context
- Match column names to question keywords ("waste", "GHG", "energy", etc.)
- This would prevent retrieving cells from different metrics

## Recommendation

**PRIMARY FINDING:** The fixes applied are in the right direction but targeting the wrong pipeline component. The `agentic_multi_tool` pipeline uses a simple calculation engine that doesn't have table-aware operand selection. The sophisticated `table_aware_grounding.py` module exists but isn't being used by the benchmark.

**ACTION:** Either:
1. Configure benchmark to use `agentic_table_aware` pipeline, OR
2. Import table-aware logic into `agentic_multi_tool` pipeline

---

**Metrics Summary (Quantitative Only):**
- Exact Match: 5.3%
- Tolerance Match (5%): 6.0%
- Relative Correctness: 35.7%
- By Operation:
  - average: 0%
  - sum: 0%
  - reduction_difference: 0%
  - All quantitative: 0%
