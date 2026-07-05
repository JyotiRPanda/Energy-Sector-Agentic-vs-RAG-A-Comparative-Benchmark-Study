# Complex-Only Benchmark Setup

## Overview

This benchmark evaluates Traditional RAG vs Agentic Multi-Tool specifically on **complex reasoning tasks**, excluding simple extractive questions. This allows us to assess whether the agentic approach provides tangible benefits for challenging questions, even if full-dataset improvements appear small.

## What is "Complex"?

Complex questions require:
- **Relational reasoning**: Entity comparisons, rankings, identification of leaders/laggards
- **Quantitative reasoning**: Numeric calculations, aggregations, percentage changes, ratios
- **Multistep reasoning**: Multi-stage operations requiring intermediate results
- **Multi-table reasoning**: Evidence integration across multiple tables

## Files

### Configuration
- **`configs/benchmark_complex_subset.yaml`** - Specifies datasets and pipelines
  - Includes 6 complex question types (no simple extractive)
  - Uses full corpus for evaluation
  - Strict mode enabled

### Script
- **`scripts/run_complex_subset_benchmark.py`** - Main benchmark runner
  - Runs both pipelines with live Azure OpenAI client
  - Generates paired comparisons
  - Computes McNemar significance tests
  - Creates detailed markdown report

### Outputs
- **`results/complex/live_summary.json`** - Aggregated metrics and statistics
- **`results/complex/live_predictions.json`** - Paired predictions for all questions
- **`results/complex/traditional_rag_live_predictions.json`** - Full RAG predictions
- **`results/complex/agentic_multi_tool_live_predictions.json`** - Full agentic predictions
- **`docs/generated/complex_subset_report.md`** - Comprehensive analysis report

## Running the Benchmark

### Prerequisites
1. Environment variables configured (`.env` file):
   ```
   PROJECT_ENDPOINT=<your-azure-endpoint>
   API_KEY=<your-azure-key>
   MODEL_DEPLOYMENT=<gpt-4o-deployment-name>
   EMBEDDING_DEPLOYMENT=<ada-deployment-name>
   ```

2. Full corpus built:
   ```bash
   python scripts/build_retrieval_corpus.py --config configs/benchmark_complex_subset.yaml
   ```

### Run Benchmark
```bash
python scripts/run_complex_subset_benchmark.py \
    --config configs/benchmark_complex_subset.yaml \
    --env-file .env \
    --output-dir results/complex \
    --report-md docs/generated/complex_subset_report.md
```

## Metrics and Analysis

### Primary Metrics
- **Exact Match Accuracy**: Strict string matching of predicted vs gold answers
- **Numeric Tolerance Match**: Numeric answers within 5% relative error (for quantitative Qs)
- **Citation Precision**: Proportion of citations that match expected evidence
- **Citation Recall**: Coverage of all expected citation sources
- **Faithfulness**: Whether answer is supported by retrieved evidence
- **Transparency**: Availability of trace steps and citations
- **Latency**: End-to-end prediction time
- **Cost**: Azure API usage cost in USD

### Paired Outcome Analysis
For each question, we compare:
- **Accuracy**: Which pipeline produced correct answer (agentic win/loss/tie)
- **Citation Quality**: Which citations were more valid
- **Numeric Tolerance**: For quantitative questions, which better matched within tolerance

### Statistical Significance
**McNemar's Test** (α=0.05) determines if observed differences are statistically significant:
- Tests only **discordant pairs** (where one pipeline won)
- Tests for **accuracy**, **citation quality**, and **numeric reasoning** separately
- Reports chi-square statistic and p-value

## Report Sections

1. **Dataset Overview** - Count of examples by question type
2. **Primary Metrics Comparison** - Side-by-side metric comparison with deltas
3. **Numeric Tolerance Matching** - Specific analysis for quantitative questions
4. **Paired Outcome Analysis** - Win/loss/tie breakdowns
5. **Statistical Significance** - McNemar test results
6. **Key Findings** - Interpretation of results
7. **Conclusion** - Summary and implications

## Interpreting Results

### High Agentic Win Rate (>60% on complex tasks)
- **Interpretation**: Agentic approach excels at complex reasoning
- **Why**: Tool usage (calculation, verification) provides advantages
- **Action**: Consider agentic as primary for enterprise queries

### McNemar Significant (p < 0.05)
- **Interpretation**: Observed differences are not random
- **Why**: Consistent advantage across multiple questions
- **Action**: Results are reliable, not due to chance

### High Numeric Tolerance but Low Exact Match
- **Interpretation**: Formatting differences, not conceptual errors
- **Why**: Different rounding, unit conversions, precision
- **Action**: Consider numeric_tolerance_match more permissive for quantitative tasks

## Comparing Full Dataset vs Complex Subset

If full-dataset shows small improvement but complex-only shows large improvement:
- **Finding**: Agentic advantage appears mainly on complex reasoning
- **Implication**: Agentic approach should be used selectively for hard questions
- **Opportunity**: Ensemble approach (route complex → agentic, simple → RAG)

## Example Runs and Outputs

See `docs/generated/` for example reports from previous benchmark runs.

## Customization

To change which questions are "complex":
1. Edit `configs/benchmark_complex_subset.yaml` to add/remove datasets
2. Modify `_is_quantitative_split()` in script to adjust categorization
3. Update `_paired_outcome_counts()` to track different metrics

## Troubleshooting

### Script fails to connect to Azure
- Check `.env` file has correct credentials
- Verify Azure OpenAI deployment names match configured values
- Test connection: `python -c "from gri_benchmark.live_clients import maybe_create_live_client; maybe_create_live_client(force=True)"`

### Report shows only ties (equal performance)
- Check that pipelines are producing different answers
- Verify citations are being tracked correctly
- Review raw predictions: `cat results/complex/live_predictions.json | jq .[0]`

### McNemar test shows as not significant
- May need more examples to achieve significance
- Check if both pipelines are performing similarly (small effect size)
- Consider focusing on specific question types instead of all complex

## Future Enhancements

Possible extensions:
1. **Hierarchical reasoning**: Add support for questions with dependency chains
2. **Domain-specific analysis**: Separate reports by question domain (emissions, energy, etc.)
3. **Error taxonomy**: Categorize where each pipeline fails differently
4. **Ablation analysis**: Remove agentic features one-by-one to measure impact
5. **Cost-efficiency frontier**: Pareto analysis of accuracy vs cost tradeoffs
