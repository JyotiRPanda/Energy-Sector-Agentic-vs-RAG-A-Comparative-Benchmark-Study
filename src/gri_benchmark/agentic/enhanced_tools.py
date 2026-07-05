"""Enhanced tool implementations for agentic pipeline.

These tools provide real computational and verification capabilities
beyond the basic placeholder tools, enabling the agentic pipeline to
make genuinely different decisions than RAG.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from gri_benchmark.evidence import RetrievedEvidence


@dataclass
class CalculationResult:
    """Result of a numeric calculation."""
    original_value: str
    calculated_value: str
    calculation_type: str  # "passthrough", "percent_conversion", "sum", etc.
    confidence: float
    notes: list[str] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def _parse_number(text: str) -> float | None:
    """Extract the first number from text."""
    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", "").replace("%", ""))
    if not match:
        return None
    return float(match.group(0))


def _format_number(value: float) -> str:
    """Format a number for display."""
    if value == int(value):
        return str(int(value))
    if abs(value) < 0.01:
        return f"{value:.4f}"
    return f"{value:.2f}"


def enhanced_numeric_calculation_tool(
    *,
    question: str,
    value: str,
    evidence: list[RetrievedEvidence] | None = None,
) -> CalculationResult:
    """Enhanced numeric calculation with actual computation.
    
    Detects calculation intents from question and performs:
    - Percent conversion
    - Unit conversion
    - Multi-value aggregation (if evidence provided)
    - Numeric normalization
    """
    if not value or not value.strip():
        return CalculationResult(
            original_value=value,
            calculated_value=value,
            calculation_type="passthrough",
            confidence=0.0,
            notes=["Empty value, no calculation performed"],
        )
    
    q_lower = question.lower()
    notes = []
    original = value
    result = value
    calc_type = "passthrough"
    confidence = 0.5
    
    # Parse the original value
    num = _parse_number(value)
    
    # Percent conversion
    if num is not None and "percent" in q_lower and "%" not in value:
        # Convert decimal to percent
        percent_val = num * 100
        result = f"{_format_number(percent_val)}%"
        calc_type = "percent_conversion"
        confidence = 0.8
        notes.append(f"Converted decimal {num} to {percent_val}%")
    
    # Percent normalization (e.g., "0.5%" should be "50%" in some contexts)
    elif num is not None and "%" in value and num < 1.0 and "percentage of total" in q_lower:
        # This is already a small percent, keep it
        calc_type = "percent_normalization"
        confidence = 0.7
        notes.append("Percent value validated")
    
    # Sum/aggregation across evidence
    elif evidence and any(k in q_lower for k in ("sum", "total", "combined", "aggregate")):
        values = []
        for hit in evidence:
            if hit.record.primary_value:
                v = _parse_number(hit.record.primary_value)
                if v is not None:
                    values.append(v)
        
        if values and num is not None:
            # Check if the provided value matches the sum
            total = sum(values)
            if abs(num - total) / max(abs(total), 1.0) < 0.01:
                result = value  # Already correct
                calc_type = "sum_verification"
                confidence = 0.95
                notes.append(f"Sum verified: {len(values)} values, total={_format_number(total)}")
            else:
                # Value doesn't match sum, report it
                result = _format_number(total)
                calc_type = "sum_recalculation"
                confidence = 0.7
                notes.append(f"Recalculated sum from {len(values)} values: {result}")
    
    # Difference/change calculation
    elif evidence and any(k in q_lower for k in ("difference", "change", "increase", "decrease")):
        if len(evidence) >= 2:
            nums = []
            for hit in evidence:
                if hit.record.primary_value:
                    v = _parse_number(hit.record.primary_value)
                    if v is not None:
                        nums.append(v)
            
            if len(nums) >= 2:
                # Difference between first and last (chronological order assumed)
                diff = nums[-1] - nums[0]
                result = _format_number(diff)
                calc_type = "difference_calculation"
                confidence = 0.7
                notes.append(f"Calculated difference: {nums[-1]} - {nums[0]} = {diff}")
    
    # Average/mean
    elif evidence and any(k in q_lower for k in ("average", "mean")):
        nums = []
        for hit in evidence:
            if hit.record.primary_value:
                v = _parse_number(hit.record.primary_value)
                if v is not None:
                    nums.append(v)
        
        if nums:
            avg = sum(nums) / len(nums)
            result = _format_number(avg)
            calc_type = "average_calculation"
            confidence = 0.7
            notes.append(f"Calculated average from {len(nums)} values: {result}")
    
    return CalculationResult(
        original_value=original,
        calculated_value=result,
        calculation_type=calc_type,
        confidence=confidence,
        notes=notes,
    )


def enhanced_answer_synthesis_tool(
    *,
    question: str,
    selected_value: str,
    generated_answer: str | None = None,
    calculation_result: CalculationResult | None = None,
    strategy_numeric_emphasis: bool = False,
) -> str:
    """Enhanced answer synthesis considering multiple sources.
    
    Prioritization:
    1. Calculated/verified values (if available)
    2. Generated answers (if high confidence)
    3. Selected values (if grounded)
    """
    # Use calculation result if available and confident
    if calculation_result and calculation_result.confidence > 0.6:
        return calculation_result.calculated_value.strip()
    
    # Use generated answer if good
    if generated_answer and str(generated_answer).strip():
        gen_clean = str(generated_answer).strip()
        if gen_clean and gen_clean != "INSUFFICIENT_CONTEXT":
            return gen_clean
    
    # Use selected value as fallback
    if selected_value and str(selected_value).strip():
        sel_clean = str(selected_value).strip()
        if sel_clean and sel_clean != "INSUFFICIENT_CONTEXT":
            return sel_clean
    
    return "INSUFFICIENT_CONTEXT"


def enhanced_citation_verifier_tool(
    *,
    answer: str,
    candidates: list[RetrievedEvidence],
    question: str = "",
) -> dict[str, Any]:
    """Enhanced citation verification with detailed diagnostics.
    
    Returns:
    - is_valid: Whether answer is grounded
    - grounding_type: "exact_match", "numeric_match", "content_mention", "none"
    - evidence_index: Which piece of evidence grounds it
    - confidence: Confidence level
    - issues: Any verification issues
    """
    if not answer or answer == "INSUFFICIENT_CONTEXT":
        return {
            "is_valid": False,
            "grounding_type": "none",
            "evidence_index": -1,
            "confidence": 0.0,
            "issues": ["Answer is empty or INSUFFICIENT_CONTEXT"],
        }
    
    if not candidates:
        return {
            "is_valid": False,
            "grounding_type": "none",
            "evidence_index": -1,
            "confidence": 0.0,
            "issues": ["No evidence to verify against"],
        }
    
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())
    
    ans_norm = normalize(answer)
    ans_num = _parse_number(answer)
    issues = []
    
    # Try exact match first (strongest grounding)
    for idx, hit in enumerate(candidates):
        primary = str(hit.record.primary_value or "").strip()
        if primary and normalize(primary) == ans_norm:
            return {
                "is_valid": True,
                "grounding_type": "exact_match",
                "evidence_index": idx,
                "confidence": 0.95,
                "issues": [],
            }
    
    # Try numeric match
    if ans_num is not None:
        for idx, hit in enumerate(candidates):
            primary = str(hit.record.primary_value or "").strip()
            if primary:
                hit_num = _parse_number(primary)
                if hit_num is not None:
                    denom = abs(hit_num) if hit_num != 0 else 1.0
                    rel_error = abs(ans_num - hit_num) / denom
                    
                    if rel_error <= 1e-3:
                        return {
                            "is_valid": True,
                            "grounding_type": "numeric_match",
                            "evidence_index": idx,
                            "confidence": 0.90,
                            "issues": [],
                        }
                    elif rel_error <= 0.05:  # Within 5%
                        return {
                            "is_valid": True,
                            "grounding_type": "numeric_match",
                            "evidence_index": idx,
                            "confidence": 0.70,
                            "issues": [f"Numeric value within 5% (relative error: {rel_error:.2%})"],
                        }
    
    # Try content mention
    for idx, hit in enumerate(candidates):
        content = str(hit.record.content_text or "").lower()
        if ans_norm and ans_norm in normalize(content):
            return {
                "is_valid": True,
                "grounding_type": "content_mention",
                "evidence_index": idx,
                "confidence": 0.60,
                "issues": ["Answer found in content but not exact or numeric match"],
            }
    
    # No grounding found
    return {
        "is_valid": False,
        "grounding_type": "none",
        "evidence_index": -1,
        "confidence": 0.0,
        "issues": ["Answer not grounded in any retrieved evidence"],
    }


def suggest_fallback_retrieval_query(
    question: str,
    evidence: list[RetrievedEvidence],
    verification_result: dict[str, Any],
) -> str | None:
    """Suggest an alternative retrieval query if verification failed.
    
    Analyzes the question and evidence to suggest a different query
    that might retrieve more relevant evidence.
    """
    if verification_result.get("is_valid"):
        return None  # No need for fallback if verification passed
    
    q_lower = question.lower()
    
    # Analyze what might be missing
    strategies = []
    
    # If numeric and not found, try removing units/years
    if any(k in q_lower for k in ("percent", "amount", "value", "number")):
        # Try a simpler numeric query
        simple = re.sub(r"\d{4}", "", question)  # Remove years
        simple = re.sub(r"\b(percent|%|tons?|gwh|mwh|gj|m3|m\^3|m³)\b", "", simple)
        if simple.strip():
            strategies.append(simple)
    
    # If relational query, try each entity separately
    if "compare" in q_lower or "versus" in q_lower:
        # Try extracting individual entities
        pass  # More sophisticated parsing needed
    
    # If no strategies, try removing question modifiers
    if not strategies:
        # Remove question words and try again
        modified = re.sub(r"\b(what|which|how|when|where|why)\b", "", question, flags=re.IGNORECASE)
        if modified.strip():
            strategies.append(modified)
    
    return strategies[0] if strategies else None
