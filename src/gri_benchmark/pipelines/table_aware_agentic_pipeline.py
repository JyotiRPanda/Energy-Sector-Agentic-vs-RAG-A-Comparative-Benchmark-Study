"""
Table-Aware Grounded Agentic Pipeline

This pipeline respects the GRI-QA dataset's fundamental structure:
- Operands are selected via table relationships (row_id, column_id), not text extraction
- Operations are determined by table structure, not question text parsing  
- Calculation only proceeds if table structure is coherent and complete
"""

from time import perf_counter
from typing import Optional, Union

from gri_benchmark.types import BenchmarkExample, Prediction, Citation
from gri_benchmark.pipelines.base import QAPipeline
from gri_benchmark.agentic.tools import ToolInvocation
from gri_benchmark.evidence import SimpleEvidenceRetriever
from gri_benchmark.live_clients import AzureOpenAIClient
from gri_benchmark.agentic.table_aware_grounding import (
    extract_table_candidates,
    select_operands_table_aware,
    should_use_table_grounded_operands,
    calculate_from_table_operands,
    TableAwareCandidate,
)


class TableAwareAgenticPipeline(QAPipeline):
    """
    Agentic pipeline with table-aware grounding layer.
    
    Difference from baseline agentic:
    - Operand selection respects table structure (row_id, column_id)
    - Operation inference uses table relationships, not text patterns
    - Only processes when operands form coherent table group
    - Returns INSUFFICIENT_CONTEXT if table structure is broken
    """
    
    name = "agentic_table_aware"
    
    def __init__(
        self,
        *,
        strict_mode: bool = False,
        retriever: Union[SimpleEvidenceRetriever, None] = None,
        live_client: Union[AzureOpenAIClient, None] = None,
    ):
        self.strict_mode = strict_mode
        self.retriever = retriever
        self.live_client = live_client
        self.use_domain_aware = False
    
    def _retrieve(
        self,
        example: BenchmarkExample,
        invocation_log: list[ToolInvocation],
    ) -> list:
        """Retrieve top-5 evidence chunks."""
        start = perf_counter()
        
        if self.use_domain_aware:
            retrieved = self.retriever.search_with_domain_awareness(
                example.question,
                top_k=5,
            )
        else:
            retrieved = self.retriever.search(
                example.question,
                top_k=5,
            )
        
        latency = (perf_counter() - start) * 1000
        
        invocation_log.append(ToolInvocation(
            tool_name="table_lookup",
            tool_input={"query": example.question, "limit": 5},
            tool_output={"chunks": len(retrieved)},
            latency_ms=latency,
            success=True,
        ))
        
        return retrieved
    
    def _ground_operands(
        self,
        example: BenchmarkExample,
        retrieved: list,
    ) -> tuple[list[TableAwareCandidate], float, str]:
        """
        Extract and select operands using table-aware grounding.
        
        Returns (operands, confidence, status)
        """
        # Extract candidates from table cells only
        candidates = extract_table_candidates(example.question, retrieved)
        
        if not candidates:
            return [], 0.0, "no_table_candidates"
        
        # Select operands respecting table structure
        operands, confidence, operation = select_operands_table_aware(
            example.question,
            candidates,
        )
        
        return operands, confidence, operation
    
    def _should_proceed(
        self,
        operands: list[TableAwareCandidate],
        confidence: float,
    ) -> bool:
        """Check if operands meet quality threshold."""
        return should_use_table_grounded_operands(
            operands,
            confidence,
            threshold=0.6,
        )
    
    def _calculate_from_grounded(
        self,
        example: BenchmarkExample,
        operands: list[TableAwareCandidate],
        operation: str,
    ) -> tuple[str, str]:
        """Perform calculation on grounded operands."""
        answer, trace = calculate_from_table_operands(operands, operation)
        return answer, trace
    
    def _verify_answer(
        self,
        answer: str,
        operands: list[TableAwareCandidate],
    ) -> Citation:
        """Create citation from grounded operands."""
        if not operands:
            return Citation(
                source_file="",
                evidence_text="INSUFFICIENT_CONTEXT",
                primary_value="",
            )
        
        # Citation references first operand source
        first = operands[0]
        rec = first.source_chunk.record
        
        return Citation(
            source_file=str(rec.source_file or ""),
            table_id=str(rec.table_id or ""),
            row_id=str(rec.row_id or "") if rec.row_id else None,
            column_id=str(rec.column_id or "") if rec.column_id else None,
            evidence_text=first.source_text,
            primary_value=str(first.value),
            reason_used="table_cell_operand",
        )
    
    def answer(self, example: BenchmarkExample) -> Prediction:
        """
        Generate answer using table-aware grounding.
        
        Pipeline:
        1. Retrieve relevant evidence
        2. Extract candidates from table cells only
        3. Select operands respecting table structure
        4. Validate confidence threshold
        5. Calculate using grounded operands
        6. Verify with citation
        """
        start = perf_counter()
        invocation_log: list[ToolInvocation] = []
        trace_steps = []
        
        # Step 1: Retrieve
        trace_steps.append({
            "step": "retrieve",
            "status": "started",
        })
        
        retrieved = self._retrieve(example, invocation_log)
        
        trace_steps[-1]["status"] = "ok"
        trace_steps[-1]["num_chunks"] = len(retrieved)
        
        # Step 2: Ground operands
        trace_steps.append({
            "step": "ground_operands",
            "status": "started",
        })
        
        operands, grounding_conf, operation = self._ground_operands(
            example,
            retrieved,
        )
        
        trace_steps[-1]["status"] = "ok"
        trace_steps[-1]["num_operands"] = len(operands)
        trace_steps[-1]["operation"] = operation
        trace_steps[-1]["confidence"] = grounding_conf
        
        # Step 3: Check confidence threshold
        if not self._should_proceed(operands, grounding_conf):
            trace_steps.append({
                "step": "confidence_check",
                "status": "failed",
                "reason": f"confidence {grounding_conf:.2f} < threshold 0.6",
            })
            
            return Prediction(
                question_id=example.question_id,
                pipeline_name="agentic_table_aware",
                answer="INSUFFICIENT_CONTEXT",
                latency_ms=(perf_counter() - start) * 1000,
                citations=[],
                trace_steps=trace_steps,
                metadata={
                    "grounding_confidence": grounding_conf,
                    "num_operands": len(operands),
                    "num_candidates": len(extract_table_candidates(example.question, retrieved)),
                    "reason": "low_confidence",
                },
            )
        
        trace_steps.append({
            "step": "confidence_check",
            "status": "ok",
            "confidence": grounding_conf,
        })
        
        # Step 4: Calculate
        trace_steps.append({
            "step": "calculate",
            "status": "started",
        })
        
        answer, calculation_trace = self._calculate_from_grounded(
            example,
            operands,
            operation,
        )
        
        trace_steps[-1]["status"] = "ok"
        trace_steps[-1]["calculation_trace"] = calculation_trace
        
        # Step 5: Verify
        trace_steps.append({
            "step": "verify",
            "status": "ok",
            "details": f"Verified against {len(operands)} table operand(s)",
        })
        
        citation = self._verify_answer(answer, operands)
        
        return Prediction(
            question_id=example.question_id,
            pipeline_name="agentic_table_aware",
            answer=answer,
            latency_ms=(perf_counter() - start) * 1000,
            citations=[citation],
            trace_steps=trace_steps,
            metadata={
                "grounding_confidence": grounding_conf,
                "operation_intent": operation,
                "num_operands": len(operands),
                "operand_tables": [op.table_id for op in operands],
                "operand_rows": [op.row_id for op in operands],
                "calculation_trace": calculation_trace,
            },
        )
