# Research Questions and Operationalization

## RQ1
Do agentic, multi-tool pipelines improve faithfulness and transparency relative to a traditional retrieval-augmented baseline on table-grounded sustainability questions?

Operational metrics:
- `faithfulness`: model-reported support score in [0,1], intended to be replaced by judge-based support checks.
- `transparency`: binary presence of both process trace and citations.

## RQ2
How do correctness, citation quality, and latency trade off between the two systems under the same benchmark and evaluation protocol?

Operational metrics:
- `exact_match` for categorical/extractive answers.
- `numeric_relative_error` for quantitative answers.
- `citation_precision` and `citation_recall`.
- `latency_ms` mean per sample.

## RQ3
Which error types dominate each pipeline, including unsupported claims, miscitations, incorrect quantitative operations, and tool-based reasoning failures?

Operational metrics:
- `error_rate.unsupported_claim`
- `error_rate.miscitation`
- `error_rate.incorrect_quantitative_operation`
- `error_rate.tool_reasoning_failure`

## Notes
Current repository provides runnable baseline instrumentation and placeholders for your actual models/tools.
Replace deterministic stubs with production pipelines to produce conclusive empirical findings.
