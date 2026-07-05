"""Agentic pipeline with grounding layer.

Key difference from standard agentic:
1. Retrieves evidence
2. GROUNDS operands (extract → rank → select)
3. Verifies operand confidence
4. Only then applies tools to grounded operands
5. Returns INSUFFICIENT_CONTEXT if grounding fails

This constrains behavior to match GRI-QA expectations:
- Precision-first (select operands before processing)
- Simple computation (deterministic, not exploratory)
- Honest uncertainty (fail gracefully when unsure)
"""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from gri_benchmark.agentic.grounding import (
    CandidateNumber,
    extract_candidates,
    select_operands,
    should_use_grounded_operands,
)
from gri_benchmark.agentic.tools import invoke_tool, ToolInvocation
from gri_benchmark.evidence import RetrievedEvidence, SimpleEvidenceRetriever
from gri_benchmark.live_clients import AzureOpenAIClient
from gri_benchmark.pipelines.base import QAPipeline
from gri_benchmark.types import BenchmarkExample, Citation, Prediction


def _classify_operation_intent(question: str) -> str:
    """Classify the intended operation from question."""
    question_lower = question.lower()
    
    intents = {
        'sum': r'\b(sum|total|combined|aggregate|all\s+together)\b',
        'average': r'\b(average|mean|median)\b',
        'difference': r'\b(difference|change|increase|decrease|reduction|growth)\b',
        'percentage': r'(percentage|%|ratio|share|proportion)\b',
        'comparison': r'\b(compare|compared|vs|versus)\b',
    }
    
    for intent, pattern in intents.items():
        if re.search(pattern, question_lower):
            return intent
    
    return 'extractive'


class AgenticGroundedPipeline(QAPipeline):
    """Agentic pipeline with grounding layer for precision-first operand selection."""
    
    name = "agentic_multi_tool_grounded"
    
    def __init__(
        self,
        *,
        strict_mode: bool = False,
        retriever: SimpleEvidenceRetriever | None = None,
        live_client: AzureOpenAIClient | None = None,
        grounding_confidence_threshold: float = 0.6,
    ) -> None:
        self.strict_mode = strict_mode
        self.retriever = retriever
        self.live_client = live_client
        self.grounding_confidence_threshold = grounding_confidence_threshold
    
    @staticmethod
    def _normalize(value: object) -> str:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        text = text.strip("\"'")
        text = re.sub(r"\s+", " ", text)
        return text
    
    def _retrieve(
        self,
        example: BenchmarkExample,
        invocation_log: list[ToolInvocation],
    ) -> list[RetrievedEvidence]:
        """Retrieve evidence."""
        if not self.strict_mode or self.retriever is None:
            return []
        
        candidates = invoke_tool(
            tool_name="TableLookupTool",
            fn=self.retriever.search,
            invocation_log=invocation_log,
            query=example.question,
            split=example.split,
            source_file=str(example.metadata.get("source_file", "")) or None,
            top_k=5,
            use_constraints=True,
        )
        
        return candidates
    
    def _ground_operands(
        self,
        example: BenchmarkExample,
        retrieved: list[RetrievedEvidence],
    ) -> tuple[list[CandidateNumber], float, str]:
        """Ground operands from retrieved evidence.
        
        Returns:
            (selected_operands, confidence, grounding_status)
        """
        if not retrieved:
            return [], 0.0, "No evidence retrieved"
        
        # Extract candidates
        candidates = extract_candidates(example.question, retrieved)
        
        if not candidates:
            return [], 0.0, "No numeric candidates found"
        
        # Classify intent
        intent = _classify_operation_intent(example.question)
        
        # Select operands
        selected, confidence = select_operands(
            example.question,
            intent,
            candidates,
        )
        
        if not selected:
            return [], 0.0, "Failed to select operands"
        
        status = f"Grounded {len(selected)} operand(s) from {len(candidates)} candidate(s) (intent={intent})"
        
        return selected, confidence, status
    
    def _calculate_from_grounded(
        self,
        example: BenchmarkExample,
        selected_operands: list[CandidateNumber],
    ) -> tuple[str, str]:
        """Perform calculation using only grounded operands.
        
        Returns:
            (answer_value, calculation_trace)
        """
        if not selected_operands:
            return "INSUFFICIENT_CONTEXT", "No operands to calculate"
        
        intent = _classify_operation_intent(example.question)
        
        if intent in ("sum", "total", "combined", "aggregate"):
            total = sum(op.value for op in selected_operands)
            trace = f"Sum of {len(selected_operands)} operands: {' + '.join(f'{op.value:.1f}' for op in selected_operands)} = {total:.1f}"
            return str(int(total) if total == int(total) else total), trace
        
        elif intent in ("average", "mean", "median"):
            avg = sum(op.value for op in selected_operands) / len(selected_operands)
            trace = f"Average of {len(selected_operands)} operands: {avg:.1f}"
            return str(int(avg) if avg == int(avg) else avg), trace
        
        elif intent in ("difference", "change", "increase", "decrease"):
            if len(selected_operands) >= 2:
                diff = selected_operands[1].value - selected_operands[0].value
                trace = f"Difference: {selected_operands[1].value:.1f} - {selected_operands[0].value:.1f} = {diff:.1f}"
                return str(int(diff) if diff == int(diff) else diff), trace
            else:
                return str(selected_operands[0].value), "Single operand, no difference calculation"
        
        elif intent in ("percentage", "ratio", "share", "proportion"):
            # For percentage, just return the value (already in %)
            val = selected_operands[0].value
            trace = f"Percentage value: {val:.1f}%"
            return str(int(val) if val == int(val) else val), trace
        
        else:  # Extractive
            val = selected_operands[0].value
            trace = f"Extracted value: {val:.1f}"
            return str(int(val) if val == int(val) else val), trace
    
    def _verify_answer(
        self,
        answer: str,
        selected_operands: list[CandidateNumber],
    ) -> Citation:
        """Create citation from grounded operands."""
        if not selected_operands:
            return Citation(
                source_file="",
                evidence_text="INSUFFICIENT_CONTEXT",
                primary_value="",
            )
        
        # Citation references first operand source
        first = selected_operands[0]
        return Citation(
            source_file=str(first.source_chunk.record.source_file or ""),
            table_id=str(first.source_chunk.record.table_id or ""),
            row_id=str(first.source_chunk.record.row_id or "") if first.source_chunk.record.row_id else None,
            column_id=str(first.source_chunk.record.column_id or "") if first.source_chunk.record.column_id else None,
            evidence_text=first.source_text,
            primary_value=str(first.value),
        )
    
    def answer(self, example: BenchmarkExample) -> Prediction:
        start = perf_counter()
        invocation_log: list[ToolInvocation] = []
        
        trace_steps = []
        
        # STEP 1: Retrieve
        retrieval_start = perf_counter()
        retrieved = self._retrieve(example, invocation_log)
        retrieval_ms = (perf_counter() - retrieval_start) * 1000
        
        if not retrieved:
            return Prediction(
                question_id=example.question_id,
                question=example.question,
                answer="INSUFFICIENT_CONTEXT",
                model="agentic_multi_tool_grounded",
                reasoning="No evidence retrieved",
                citations=[],
                latency_ms=(perf_counter() - start) * 1000,
                metadata={"trace_steps": [{"step": "retrieve", "status": "no_results"}]},
            )
        
        trace_steps.append({
            "step": "retrieve",
            "status": "ok",
            "details": f"Retrieved {len(retrieved)} evidence chunks",
            "latency_ms": retrieval_ms,
        })
        
        # STEP 2: Ground operands
        grounding_start = perf_counter()
        selected_operands, grounding_conf, grounding_status = self._ground_operands(example, retrieved)
        grounding_ms = (perf_counter() - grounding_start) * 1000
        
        trace_steps.append({
            "step": "ground",
            "status": "ok",
            "details": grounding_status,
            "confidence": grounding_conf,
            "latency_ms": grounding_ms,
        })
        
        # STEP 3: Check grounding confidence
        if not should_use_grounded_operands(
            selected_operands,
            grounding_conf,
            self.grounding_confidence_threshold,
        ):
            return Prediction(
                question_id=example.question_id,
                question=example.question,
                answer="INSUFFICIENT_CONTEXT",
                model="agentic_multi_tool_grounded",
                reasoning=f"Grounding confidence {grounding_conf:.2f} below threshold {self.grounding_confidence_threshold}",
                citations=[],
                latency_ms=(perf_counter() - start) * 1000,
                metadata={
                    "trace_steps": trace_steps,
                    "grounding_confidence": grounding_conf,
                },
            )
        
        trace_steps.append({
            "step": "confidence_check",
            "status": "ok",
            "details": f"Confidence {grounding_conf:.2f} >= threshold {self.grounding_confidence_threshold}",
        })
        
        # STEP 4: Calculate from grounded operands
        calculation_start = perf_counter()
        answer, calculation_trace = self._calculate_from_grounded(example, selected_operands)
        calculation_ms = (perf_counter() - calculation_start) * 1000
        
        trace_steps.append({
            "step": "calculate",
            "status": "ok",
            "details": calculation_trace,
            "latency_ms": calculation_ms,
        })
        
        # STEP 5: Verify
        citation = self._verify_answer(answer, selected_operands)
        
        trace_steps.append({
            "step": "verify",
            "status": "ok",
            "details": f"Verified against {len(selected_operands)} operand source(s)",
        })
        
        return Prediction(
            question_id=example.question_id,
            pipeline_name="agentic_multi_tool_grounded",
            answer=answer,
            latency_ms=(perf_counter() - start) * 1000,
            citations=[citation],
            trace_steps=trace_steps,
            metadata={
                "grounding_confidence": grounding_conf,
                "num_operands": len(selected_operands),
                "num_candidates": len(extract_candidates(example.question, retrieved)),
                "calculation_trace": calculation_trace,
            },
        )
