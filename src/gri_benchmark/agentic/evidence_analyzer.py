"""Evidence analysis and answer verification for agentic pipeline.

Provides mechanisms to:
- Analyze evidence sufficiency before synthesis
- Verify answer correctness after synthesis
- Support retry/fallback decisions
- Ground citations in verified evidence
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from gri_benchmark.evidence import RetrievedEvidence


@dataclass
class EvidenceSufficiencyReport:
    """Assessment of whether evidence adequately supports answering a question."""
    is_sufficient: bool
    confidence: float  # 0.0 to 1.0
    coverage_issues: list[str] = field(default_factory=list)
    supported_aspects: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    year_match: bool = True
    unit_match: bool = True
    table_diversity: int = 1  # Number of distinct tables in evidence
    numeric_precision: float = 1.0  # 0.0 if precision is weak
    has_primary_value: bool = False


@dataclass
class AnswerVerificationResult:
    """Result of verifying an answer against evidence."""
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    grounding_type: str  # "exact_match", "numeric_match", "content_mention", "none"
    grounding_evidence_indices: list[int] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    is_numeric: bool = False
    numeric_value: float | None = None
    relative_error: float | None = None


def _parse_number(text: str) -> float | None:
    """Extract the first number from text."""
    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if not match:
        return None
    return float(match.group(0))


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_question_intent(question: str) -> set[str]:
    """Extract intent keywords from question."""
    q_lower = question.lower()
    intents = set()
    
    # Quantitative intents
    if any(k in q_lower for k in ("sum", "total", "combined", "aggregate")):
        intents.add("sum")
    if any(k in q_lower for k in ("average", "mean", "typical")):
        intents.add("average")
    if any(k in q_lower for k in ("percent", "percentage", "fraction", "proportion")):
        intents.add("percent")
    if any(k in q_lower for k in ("difference", "change", "increase", "decrease", "reduction")):
        intents.add("difference")
    if any(k in q_lower for k in ("ratio", "compared", "versus")):
        intents.add("ratio")
    
    # Relational intents
    if any(k in q_lower for k in ("which company", "which organization", "which entity")):
        intents.add("entity_selection")
    if any(k in q_lower for k in ("higher", "lower", "maximum", "minimum", "maximum", "best", "worst")):
        intents.add("superlative")
    if any(k in q_lower for k in ("rank", "ranking", "order", "sort")):
        intents.add("ranking")
    
    # Multi-step intents
    if any(k in q_lower for k in ("first", "then", "after", "before", "step")):
        intents.add("multistep")
    if any(k in q_lower for k in ("across", "between", "combine")):
        intents.add("multi_table")
    
    return intents


def analyze_evidence_sufficiency(
    question: str,
    candidates: list[RetrievedEvidence],
    query_type: str,
) -> EvidenceSufficiencyReport:
    """Analyze whether retrieved evidence is sufficient to answer question.
    
    Returns detailed report for retry/fallback decisions.
    """
    if not candidates:
        return EvidenceSufficiencyReport(
            is_sufficient=False,
            confidence=0.0,
            coverage_issues=["No evidence retrieved"],
        )
    
    q_intents = _extract_question_intent(question)
    q_lower = question.lower()
    
    # Track aspects from evidence
    supported_aspects = []
    missing_aspects = list(q_intents)
    coverage_issues = []
    
    # Check year/unit consistency
    years_in_evidence = set()
    units_in_evidence = set()
    tables_in_evidence = set()
    numeric_values_found = []
    has_primary = False
    
    for hit in candidates:
        rec = hit.record
        
        # Track primary values
        if rec.primary_value:
            has_primary = True
            num = _parse_number(rec.primary_value)
            if num is not None:
                numeric_values_found.append(num)
        
        # Track metadata
        years_in_evidence.update(rec.years)
        units_in_evidence.update(rec.units)
        tables_in_evidence.add(rec.table_id or "")
        
        # Check intent match
        for intent in q_intents:
            if intent in rec.intents:
                if intent in missing_aspects:
                    missing_aspects.remove(intent)
                if intent not in supported_aspects:
                    supported_aspects.append(intent)
    
    # Year matching
    year_match = True
    expected_years = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
    if expected_years and not (expected_years & years_in_evidence):
        year_match = False
        coverage_issues.append(f"No evidence for years {expected_years}")
    
    # Unit matching  
    unit_match = True
    q_units = set(re.findall(r"\b(gwh|mwh|gj|tons?|t\b|m3|m\^3|m³|%|percent)\b", q_lower))
    if q_units and not (q_units & units_in_evidence):
        unit_match = False
        coverage_issues.append(f"No evidence for units {q_units}")
    
    # Check numeric precision
    numeric_precision = 1.0
    if numeric_values_found:
        # Check for variation (should be consistent for same metric)
        if len(numeric_values_found) > 1:
            min_val = min(numeric_values_found)
            max_val = max(numeric_values_found)
            if min_val > 0:
                rel_variation = (max_val - min_val) / min_val
                if rel_variation > 0.1:
                    numeric_precision = 0.7  # Soft warning on inconsistency
                    coverage_issues.append("Numeric values vary across evidence")
    
    # Determine sufficiency
    table_diversity = len(tables_in_evidence)
    num_missing = len(missing_aspects)
    
    # Criteria:
    # - Must have at least 1 candidate
    # - Year/unit constraints help
    # - Primary values help
    # - Missing intents reduce confidence
    base_score = 0.5 if candidates else 0.0
    
    if has_primary:
        base_score += 0.3
    if year_match:
        base_score += 0.1
    if unit_match:
        base_score += 0.1
    
    penalty = num_missing * 0.15  # Each unmatched intent reduces score
    confidence = max(0.0, min(1.0, base_score - penalty))
    
    # Sufficient if confidence > 0.6, or query is simple extractive
    is_sufficient = confidence > 0.6 or (query_type == "extractive" and candidates)
    
    return EvidenceSufficiencyReport(
        is_sufficient=is_sufficient,
        confidence=confidence,
        coverage_issues=coverage_issues,
        supported_aspects=supported_aspects,
        missing_aspects=missing_aspects,
        year_match=year_match,
        unit_match=unit_match,
        table_diversity=table_diversity,
        numeric_precision=numeric_precision,
        has_primary_value=has_primary,
    )


def verify_answer(
    answer: str,
    candidates: list[RetrievedEvidence],
    question: str,
) -> AnswerVerificationResult:
    """Verify that an answer is grounded in retrieved evidence.
    
    Multi-dimensional verification: exact match, numeric match, content mention.
    Returns grounding type and which evidence supports the answer.
    """
    if not answer or answer == "INSUFFICIENT_CONTEXT":
        return AnswerVerificationResult(
            is_valid=False,
            confidence=0.0,
            grounding_type="none",
            issues=["Answer is empty or INSUFFICIENT_CONTEXT"],
        )
    
    if not candidates:
        return AnswerVerificationResult(
            is_valid=False,
            confidence=0.0,
            grounding_type="none",
            issues=["No evidence available to verify against"],
        )
    
    ans_norm = _normalize_text(answer)
    ans_num = _parse_number(answer)
    issues = []
    grounding_indices = []
    best_grounding = "none"
    best_confidence = 0.0
    
    # Try to ground answer in each piece of evidence
    for idx, hit in enumerate(candidates):
        rec = hit.record
        primary = str(rec.primary_value or "").strip()
        content = str(rec.content_text or "").lower()
        
        # Exact match check
        if primary and _normalize_text(primary) == ans_norm:
            grounding_indices.append(idx)
            best_grounding = "exact_match"
            best_confidence = max(best_confidence, 0.95)
            continue
        
        # Numeric match check
        if ans_num is not None and primary:
            hit_num = _parse_number(primary)
            if hit_num is not None:
                denom = abs(hit_num) if hit_num != 0 else 1.0
                rel_error = abs(ans_num - hit_num) / denom
                if rel_error <= 1e-3:  # Allow tiny floating point errors
                    grounding_indices.append(idx)
                    if best_grounding != "exact_match":
                        best_grounding = "numeric_match"
                        best_confidence = max(best_confidence, 0.90)
                    continue
                elif rel_error <= 0.05:  # Within 5%
                    grounding_indices.append(idx)
                    if best_grounding not in ("exact_match", "numeric_match"):
                        best_grounding = "numeric_match"
                        best_confidence = max(best_confidence, 0.70)
        
        # Content mention check
        if ans_norm and ans_norm in _normalize_text(content):
            grounding_indices.append(idx)
            if best_grounding == "none":
                best_grounding = "content_mention"
                best_confidence = max(best_confidence, 0.60)
    
    if not grounding_indices:
        issues.append("Answer not grounded in any retrieved evidence")
    
    is_valid = best_grounding != "none"
    
    return AnswerVerificationResult(
        is_valid=is_valid,
        confidence=best_confidence,
        grounding_type=best_grounding,
        grounding_evidence_indices=grounding_indices,
        issues=issues,
        is_numeric=ans_num is not None,
        numeric_value=ans_num,
    )


def select_grounded_evidence(
    candidates: list[RetrievedEvidence],
    verification: AnswerVerificationResult,
    fallback_to_top: bool = True,
) -> RetrievedEvidence | None:
    """Select evidence that actually grounds the answer.
    
    Uses verification result to pick citation evidence, not just top-k.
    Falls back to top candidate if needed.
    """
    if not candidates:
        return None
    
    if verification.grounding_evidence_indices:
        # Return the highest-scoring grounded evidence
        grounded = [candidates[i] for i in verification.grounding_evidence_indices]
        # Sort by score (higher is better)
        grounded.sort(key=lambda x: x.score, reverse=True)
        return grounded[0]
    
    # Fallback to top if requested
    if fallback_to_top:
        return candidates[0]
    
    return None
