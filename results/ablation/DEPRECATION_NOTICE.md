# ⚠️ ABLATION RESULTS DEPRECATION NOTICE

**Date**: 2026-07-05
**Status**: SUPERSEDED - Regenerated with corrected metrics

## Issue

The ablation results in `agentic_tools/` directory were generated with an **outdated metric formula** before the symmetric NRE formula was implemented.

### Metric Formula Mismatch

- **Old Formula** (used in these results): `|p-g|/|g|` (unbounded)
- **New Formula** (current code): `(2|p-g|)/(|g|+|p|)` (bounded to [0, 2])

### Impact on Metrics

| Metric | Old Value | New Value | Change |
|--------|-----------|-----------|--------|
| agentic_multi_tool NRE | 79394.68 ❌ | 0.7257 ✓ | **-88,000%** |
| agentic_multi_tool_no_tools NRE | 79394.68 ❌ | 0.9470 ✓ | **-88,000%** |

The NRE values were completely incorrect due to unbounded formula producing extreme outlier values.

## Resolution

### ✓ New Valid Results
Location: `results/ablation/agentic_tools/summary.json` (regenerated 2026-07-05)

**Updated Metrics:**
- `agentic_multi_tool`: NRE=0.7257, EM=0.4967, Citation Precision=0.5421, Faithfulness=0.6962
- `agentic_multi_tool_no_tools`: NRE=0.9470, EM=0.3575, Citation Precision=0.5259, Faithfulness=0.7091

### ✗ Deprecated Results
- Prediction files (now stale): `agentic_multi_tool_predictions.json`, `agentic_multi_tool_no_tools_predictions.json`
  - **These were REGENERATED** with same logic, only metrics were recalculated, so predictions are still valid

## For Thesis Usage

**Use the regenerated `summary.json`** - metrics now match the symmetric formula in codebase.

All evaluation scores reflect the correct, bounded NRE formula: `(2|p-g|)/(|g|+|p|)`

See [metrics.py](../../src/gri_benchmark/evaluation/metrics.py#L99) for implementation.
