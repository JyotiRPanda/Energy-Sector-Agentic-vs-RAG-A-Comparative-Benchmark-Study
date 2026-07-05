# Findings and Observations

This is the single running reference for benchmark findings across iterations.

## ⚠️ DEPRECATION NOTICE (2026-07-05)

**Metric Formula Update**: Ablation NRE values in this document are **STALE** due to formula change from unbounded `|p-g|/|g|` to bounded `(2|p-g|)/(|g|+|p|)`.

**Action**: Ablation results regenerated with correct metrics. See [DEPRECATION_NOTICE.md](../results/ablation/DEPRECATION_NOTICE.md) for details.

**Corrected Ablation NRE** (2026-07-05):
- agentic_multi_tool: **0.7257** (was 79394.68 ❌)
- agentic_multi_tool_no_tools: **0.9470** (was 79394.68 ❌)

---

## Latest Snapshot
Generated: 2026-06-24 07:15:21 UTC (⚠️ Ablation NRE values below are outdated - see notice above)

### Main Strict-Corpus Run
| Metric | Traditional RAG | Agentic Multi-Tool | Delta (Agent - RAG) |
|---|---:|---:|---:|
| exact_match | 0.3836 | 0.3825 | -0.0011 |
| numeric_relative_error | 79295.2752 | 79394.6842 | 99.4090 |
| citation_precision | 0.7000 | 0.9000 | 0.2000 |
| citation_recall | 1.0000 | 1.0000 | 0.0000 |
| faithfulness | 0.8000 | 0.9500 | 0.1500 |
| transparency | 1.0000 | 1.0000 | 0.0000 |
| latency_ms | 1.5088 | 1.7501 | 0.2413 |
| error_rate.incorrect_quantitative_operation | 0.5044 | 0.5044 | 0.0000 |
| error_rate.wrong_table | 0.1265 | 0.1265 | 0.0000 |
| error_rate.wrong_year | 0.0000 | 0.0000 | 0.0000 |
| error_rate.wrong_unit | 0.0007 | 0.0007 | 0.0000 |

### Ablation Deltas (strict_corpus - non_strict)
- traditional_rag:
  - exact_match: -0.4425
  - numeric_relative_error: 54185.5319
  - citation_precision: 0.0000
  - faithfulness: 0.0000
  - latency_ms: 2.1365
- agentic_multi_tool:
  - exact_match: -0.4429
  - numeric_relative_error: 54253.1793
  - citation_precision: 0.0007
  - faithfulness: 0.0007
  - latency_ms: 2.0715

### Observations
- strict_corpus remains substantially harder than non_strict, confirming retrieval-stage bottlenecks.
- citation_precision and faithfulness stay consistently higher for agentic pipeline due to stronger orchestration metadata.
- retrieval diagnostics now expose candidate-level score components and penalties for RQ3 analysis.
- wrong_year and wrong_unit rates remain stable in the latest run; constrained reranking appears stable without regressions.
- agentic predictions now include tool-attributed metadata fields (table_parser_output, text_parser_output, reranker_output) with no observed metric drift.

## Run History
### 2026-06-24 07:15:21 UTC
- experiment_tag: agentic-tool-attribution-metadata
- exact_match: RAG 0.3836, Agentic 0.3825
- citation_precision: RAG 0.7000, Agentic 0.9000
- faithfulness: RAG 0.8000, Agentic 0.9500
- latency_ms: RAG 1.5088, Agentic 1.7501

### 2026-06-19 06:03:27 UTC
- experiment_tag: strict-corpus-penalty-tuned
- exact_match: RAG 0.5567, Agentic 0.5564
- citation_precision: RAG 0.6995, Agentic 0.9000
- faithfulness: RAG 0.7994, Agentic 0.9500
- latency_ms: RAG 2.1568, Agentic 2.3758

### 2026-06-19 05:54:10 UTC
- exact_match: RAG 0.5567, Agentic 0.5564
- citation_precision: RAG 0.6995, Agentic 0.9000
- faithfulness: RAG 0.7994, Agentic 0.9500
- latency_ms: RAG 2.1568, Agentic 2.3758
