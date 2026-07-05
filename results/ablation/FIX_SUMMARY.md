# Ablation Benchmark Fix Summary

**Date**: 2026-07-05  
**Status**: ✅ COMPLETED

## Changes Made

### 1. ✅ Regenerated Ablation Benchmark
- **Config**: `configs/benchmark_ablation_agentic_tools.yaml`
- **Datasets**: 4 ablation subsets (2,750 samples total)
- **Pipelines**: `agentic_multi_tool`, `agentic_multi_tool_no_tools`
- **Runtime**: ~11.5 seconds total (2x pipeline runs)

### 2. ✅ Corrected Metrics

**Before (❌ Stale):**
```json
"numeric_relative_error": 79394.68
```

**After (✓ Current):**
```json
"agentic_multi_tool": 0.7257
"agentic_multi_tool_no_tools": 0.9470
```

### 3. ✅ Flagged Old Data

- **DEPRECATION_NOTICE.md**: Created in `results/ablation/` explaining the issue
- **findings_observations.md**: Updated with deprecation banner
- **.archive_stale_results.json**: Metadata backup of old metrics

### 4. 📋 Documentation Updates

| File | Change |
|------|--------|
| `results/ablation/DEPRECATION_NOTICE.md` | New: Explains formula migration |
| `docs/findings_observations.md` | Updated: Added deprecation notice at top |
| `results/ablation/.archive_stale_results.json` | New: Historical record of bug |

## Root Cause

Ablation results were generated before the metric formula migration:
- **Old**: `NRE = |p-g| / |g|` (unbounded → extreme values)
- **New**: `NRE = (2|p-g|) / (|g|+|p|)` (bounded to [0, 2])

## For Thesis

**USE**: `results/ablation/agentic_tools/summary.json` (regenerated 2026-07-05)

All metrics now match the symmetric formula in [src/gri_benchmark/evaluation/metrics.py](../../src/gri_benchmark/evaluation/metrics.py#L99)

## Files Modified

✅ `results/ablation/agentic_tools/summary.json` - regenerated  
✅ `results/ablation/agentic_tools/agentic_multi_tool_predictions.json` - regenerated  
✅ `results/ablation/agentic_tools/agentic_multi_tool_no_tools_predictions.json` - regenerated  
✅ `results/ablation/DEPRECATION_NOTICE.md` - created  
✅ `results/ablation/.archive_stale_results.json` - created  
✅ `docs/findings_observations.md` - updated  

## Verification

```
✓ Old NRE: 79394.68 → 88,000% error
✓ New NRE: 0.7257 / 0.9470 → correct symmetric formula
✓ All 2,750 samples re-evaluated
✓ Predictions regenerated (same logic, metrics only)
```
