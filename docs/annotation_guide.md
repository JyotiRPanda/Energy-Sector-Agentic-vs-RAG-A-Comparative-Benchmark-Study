# Error Annotation Guide

## Label Definitions
- unsupported_claim: answer statement not grounded in cited table evidence.
- miscitation: citation points to wrong table/cell or does not support the claim.
- incorrect_quantitative_operation: arithmetic/comparison/aggregation error.
- tool_reasoning_failure: planning or tool invocation failure causing wrong output.

## Annotation Steps
1. Read question and reference answer.
2. Inspect model answer and citations.
3. Verify evidence in table source.
4. Assign one or more error labels.
5. Add short rationale in free text.

## Suggested Fields
- question_id
- pipeline_name
- assigned_labels
- rationale
- annotator_id
- adjudicated_label_set
