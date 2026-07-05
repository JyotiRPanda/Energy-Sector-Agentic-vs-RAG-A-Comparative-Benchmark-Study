"""Enhanced question-type routing for agentic pipeline.

Routes questions to different strategies based on type, with concrete behavioral differences:
- Quantitative: calculation-first, numeric verification
- Relational: multi-candidate comparison
- Multi-step: cross-table evidence gathering
- Extractive: simple but verified extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from gri_benchmark.evidence import RetrievedEvidence


QueryType = Literal["extractive", "relational", "quantitative", "multi_step"]


@dataclass(frozen=True)
class RoutingStrategy:
    """Strategy for processing a specific query type."""
    query_type: QueryType
    priority_dimensions: list[str]  # Which evidence dimensions to prioritize
    min_evidence_required: int  # Minimum candidates needed
    allow_partial_answer: bool  # Whether to return partial/incomplete answers
    require_verification: bool  # Whether answer verification is mandatory
    allow_retry_retrieval: bool  # Whether to allow fallback retrieval
    citation_mode: str  # "grounded", "top_k", "verified"
    numeric_emphasis: bool  # Emphasize numeric values in synthesis
    multi_table_ok: bool  # Whether evidence can span multiple tables


def classify_query_type_enhanced(question: str, split: str) -> QueryType:
    """Enhanced query classification with multiple signals."""
    q = question.lower()
    s = split.lower()
    
    # Strong split-based signals
    if "multi" in s or "multistep" in s:
        return "multi_step"
    if "rel" in s:
        return "relational"
    if "quant" in s:
        return "quantitative"
    if "extract" in s or "extractive" in s:
        return "extractive"
    
    # Question-based signals with increasing strength
    quantitative_markers = ["sum", "total", "combined", "aggregate", "average", "mean", 
                           "percent", "percentage", "fraction", "ratio", "proportion", 
                           "difference", "increase", "decrease", "reduction", "change"]
    relational_markers = ["which company", "which organization", "which entity", 
                         "compare", "higher", "lower", "maximum", "minimum", 
                         "best", "worst", "rank", "ranking", "versus"]
    multistep_markers = ["first", "then", "after", "before", "step", "across tables", 
                        "multiple tables", "combine", "between"]
    
    # Check for strong markers (appears multiple times)
    quantitative_count = sum(1 for m in quantitative_markers if m in q)
    relational_count = sum(1 for m in relational_markers if m in q)
    multistep_count = sum(1 for m in multistep_markers if m in q)
    
    # Return based on strongest signal
    if multistep_count >= 1:
        return "multi_step"
    if relational_count >= 1:
        return "relational"
    if quantitative_count >= 1:
        return "quantitative"
    
    return "extractive"


def get_routing_strategy(query_type: QueryType) -> RoutingStrategy:
    """Get the processing strategy for a query type."""
    strategies: dict[QueryType, RoutingStrategy] = {
        "extractive": RoutingStrategy(
            query_type="extractive",
            priority_dimensions=["primary_value", "content_match"],
            min_evidence_required=1,
            allow_partial_answer=False,
            require_verification=False,  # Extractive is usually clear-cut
            allow_retry_retrieval=False,  # One pass is sufficient
            citation_mode="top_k",
            numeric_emphasis=False,
            multi_table_ok=False,
        ),
        "quantitative": RoutingStrategy(
            query_type="quantitative",
            priority_dimensions=["numeric_value", "units", "years"],
            min_evidence_required=1,
            allow_partial_answer=False,
            require_verification=True,  # Numeric answers must verify
            allow_retry_retrieval=True,  # May need fallback for precision
            citation_mode="grounded",  # Use evidence that grounds numeric answer
            numeric_emphasis=True,  # Prioritize numeric values
            multi_table_ok=False,  # Usually single metric
        ),
        "relational": RoutingStrategy(
            query_type="relational",
            priority_dimensions=["entity_field", "comparative_values", "year_consistency"],
            min_evidence_required=2,  # Need at least 2 candidates to compare
            allow_partial_answer=False,
            require_verification=True,  # Comparison must be valid
            allow_retry_retrieval=True,  # May need more candidates
            citation_mode="verified",  # Use multiple sources for comparison
            numeric_emphasis=False,
            multi_table_ok=True,  # Can compare across tables
        ),
        "multi_step": RoutingStrategy(
            query_type="multi_step",
            priority_dimensions=["step_sequence", "multi_table_support"],
            min_evidence_required=3,  # Need multiple sources for steps
            allow_partial_answer=True,  # May complete partial steps
            require_verification=True,  # Each step must verify
            allow_retry_retrieval=True,  # May need additional evidence
            citation_mode="grounded",  # Each step grounded in evidence
            numeric_emphasis=False,
            multi_table_ok=True,  # Cross-table is expected
        ),
    }
    return strategies[query_type]


def select_candidates_by_strategy(
    candidates: list[RetrievedEvidence],
    strategy: RoutingStrategy,
    question: str,
) -> list[RetrievedEvidence]:
    """Select and reorder candidates according to routing strategy.
    
    Different strategies prioritize different evidence dimensions.
    """
    if not candidates:
        return []
    
    def score_candidate(hit: RetrievedEvidence) -> float:
        """Score a candidate based on strategy priorities."""
        score = hit.score  # Base retrieval score
        
        for dimension in strategy.priority_dimensions:
            if dimension == "primary_value":
                if hit.record.primary_value:
                    score += 0.1
            
            elif dimension == "numeric_value":
                if hit.record.primary_value:
                    num = re.search(r"[-+]?\d*\.?\d+", hit.record.primary_value)
                    if num:
                        score += 0.15  # Boost numeric values
            
            elif dimension == "units":
                if hit.record.units:
                    score += 0.1
            
            elif dimension == "years":
                if hit.record.years:
                    score += 0.1
            
            elif dimension == "entity_field":
                # For relational queries, non-numeric values are good
                if hit.record.primary_value:
                    num = re.search(r"[-+]?\d*\.?\d+", hit.record.primary_value)
                    if not num:  # Not numeric = probably entity name
                        score += 0.1
            
            elif dimension == "comparative_values":
                # Multiple candidates with comparable values help comparison
                if hit.record.primary_value:
                    score += 0.05
            
            elif dimension == "content_match":
                # Strong content match
                if hit.record.content_text:
                    score += 0.05
        
        # Multi-table boost if allowed
        if strategy.multi_table_ok and hit.record.table_id:
            # Slightly prefer diverse tables (but this is applied per-candidate)
            pass
        
        return score
    
    # Sort by strategy-aware score
    scored = [(hit, score_candidate(hit)) for hit in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Return top candidates, ensuring at least min_evidence_required
    result = [hit for hit, _ in scored]
    return result[:max(strategy.min_evidence_required, len(result))]


def should_retry_retrieval(
    strategy: RoutingStrategy,
    sufficiency_confidence: float,
    verification_confidence: float,
    answer: str,
) -> bool:
    """Decide whether to retry retrieval based on strategy and current results.
    
    Retry if:
    - Strategy allows it
    - Confidence too low
    - Answer is empty or insufficient
    """
    if not strategy.allow_retry_retrieval:
        return False
    
    if not answer or answer == "INSUFFICIENT_CONTEXT":
        return True
    
    # For strict strategies, require higher confidence
    if strategy.require_verification and verification_confidence < 0.5:
        return True
    
    if sufficiency_confidence < 0.4:
        return True
    
    return False
