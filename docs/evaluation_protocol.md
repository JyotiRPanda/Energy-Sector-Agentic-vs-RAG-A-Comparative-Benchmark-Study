# Evaluation Protocol

## Fair Comparison Controls
- Same benchmark splits and same question set for both systems.
- Same model family when possible for generation components.
- Fixed random seeds and deterministic tool settings where available.
- Latency measured end-to-end per question.

## Reporting Requirements
- Report all metrics by pipeline and by question subset.
- Include confidence intervals through bootstrap resampling.
- Publish run config and code commit hash with each result table.

## Error Analysis Procedure
- Export per-question predictions and traces.
- Apply taxonomy rules for initial labels.
- Perform manual adjudication on a stratified sample.
- Report inter-annotator agreement for manually reviewed subset.

## Reproducibility Checklist
- Python version and dependency lock.
- Exact dataset manifest used for runs.
- Pipeline configuration for retriever/tool settings.
- Hardware/runtime details for latency interpretation.
