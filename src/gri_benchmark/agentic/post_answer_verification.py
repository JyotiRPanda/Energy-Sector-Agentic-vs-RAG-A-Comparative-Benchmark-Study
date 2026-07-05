"""Post-answer verification for agentic multi-tool pipeline.

Verifies that produced answers are supported by retrieved evidence and can retry
if initial answers are unsupported.
"""

import re
from typing import Any, Optional, Tuple


def verify_answer(
    answer: str,
    retrieved_hits: list,
    question: str,
    strategy: str,
    gold_answer: Optional[str] = None,
) -> dict[str, Any]:
    """Verify that produced answer is supported by retrieved evidence.
    
    Args:
        answer: The generated/synthesized answer
        retrieved_hits: List of RetrievedEvidence objects from retrieval
        question: Original question
        strategy: Question routing strategy
        gold_answer: Optional gold answer for development/debugging
    
    Returns:
        {
            "verified": bool,
            "support_level": "supported" | "partially_supported" | "unsupported",
            "verification_reason": string explanation,
            "used_evidence_ids": list of record_ids that support answer,
            "confidence": float 0.0-1.0,
            "needs_retry": bool indicating if retry recommended,
            "retry_recommendation": string explaining retry strategy
        }
    """
    if not answer or answer == "INSUFFICIENT_CONTEXT":
        return {
            "verified": False,
            "support_level": "unsupported",
            "verification_reason": "No answer produced",
            "used_evidence_ids": [],
            "confidence": 0.0,
            "needs_retry": False,
            "retry_recommendation": "Expand retrieval query"
        }
    
    if not retrieved_hits:
        return {
            "verified": False,
            "support_level": "unsupported",
            "verification_reason": "No evidence available for verification",
            "used_evidence_ids": [],
            "confidence": 0.0,
            "needs_retry": True,
            "retry_recommendation": "Retry retrieval with expanded query"
        }

    expected_shape = _infer_expected_answer_shape(question)
    if expected_shape == "boolean" and not _is_boolean_answer(answer):
        return {
            "verified": False,
            "support_level": "unsupported",
            "verification_reason": "Question expects boolean answer (yes/no)",
            "used_evidence_ids": [],
            "confidence": 0.0,
            "needs_retry": True,
            "retry_recommendation": "Retry with comparative evidence and boolean synthesis"
        }
    if expected_shape == "multi_value" and not _is_multi_value_answer(answer):
        return {
            "verified": False,
            "support_level": "unsupported",
            "verification_reason": "Question expects multiple ordered values",
            "used_evidence_ids": [],
            "confidence": 0.0,
            "needs_retry": True,
            "retry_recommendation": "Retry with ranking/order-aware extraction"
        }
    
    # Check if answer appears in evidence
    exact_match, match_ids = _check_answer_in_evidence(answer, retrieved_hits)
    if exact_match:
        return {
            "verified": True,
            "support_level": "supported",
            "verification_reason": f"Answer found in {len(match_ids)} evidence record(s)",
            "used_evidence_ids": match_ids,
            "confidence": 0.95,
            "needs_retry": False,
            "retry_recommendation": ""
        }
    
    # For numeric answers, check if matches computed value within tolerance
    if _is_numeric_answer(answer):
        numeric_tolerance_pct = 1.0 if strategy in ("extractive_lookup", "relational_comparison") else 5.0
        numeric_match, match_ids, tolerance_pct = _check_numeric_match(
            answer, retrieved_hits, question, strategy, tolerance_pct=numeric_tolerance_pct
        )
        if numeric_match:
            return {
                "verified": True,
                "support_level": "supported",
                "verification_reason": (
                    f"Answer matches numeric evidence within {tolerance_pct}% tolerance"
                ),
                "used_evidence_ids": match_ids,
                "confidence": 0.85,
                "needs_retry": False,
                "retry_recommendation": ""
            }
    
    # Check if answer could be partially supported (key terms present)
    partial_support, partial_ids, partial_reason = _check_partial_support(
        answer, retrieved_hits
    )
    if partial_support:
        return {
            "verified": True,
            "support_level": "partially_supported",
            "verification_reason": partial_reason,
            "used_evidence_ids": partial_ids,
            "confidence": 0.65,
            "needs_retry": False,
            "retry_recommendation": ""
        }
    
    # Answer not supported - recommend retry
    return {
        "verified": False,
        "support_level": "unsupported",
        "verification_reason": "Answer not found in any retrieved evidence",
        "used_evidence_ids": [],
        "confidence": 0.0,
        "needs_retry": True,
        "retry_recommendation": _get_retry_recommendation(strategy, retrieved_hits)
    }


def _check_answer_in_evidence(answer: str, hits: list) -> Tuple[bool, list]:
    """Check if answer appears verbatim or closely in evidence.
    
    Returns:
        (found: bool, list of matching record_ids)
    """
    answer_lower = str(answer).lower().strip()
    if not answer_lower or len(answer_lower) < 2:
        return False, []
    
    is_numeric_answer = _is_numeric_answer(answer)
    matching_ids = []
    
    for hit in hits:
        record = hit.record
        
        # Check primary value - exact match only
        primary_val = str(getattr(record, "primary_value", "")).lower().strip()
        if primary_val and primary_val == answer_lower:
            matching_ids.append(record.record_id)
            continue
        
        # Check content text
        content = str(getattr(record, "content_text", "")).lower()
        if is_numeric_answer:
            try:
                answer_num = float(answer_lower.replace(",", ""))
                for m in re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?", content):
                    token_num = float(m.group(0).replace(",", ""))
                    if abs(token_num - answer_num) <= 1e-6:
                        matching_ids.append(record.record_id)
                        break
                if matching_ids and matching_ids[-1] == record.record_id:
                    continue
            except (ValueError, TypeError):
                pass
        elif answer_lower in content:
            matching_ids.append(record.record_id)
            continue
        
        # For non-numeric answers, try fuzzy match
        if not is_numeric_answer and _fuzzy_match(answer_lower, primary_val):
            matching_ids.append(record.record_id)
    
    return len(matching_ids) > 0, matching_ids


def _check_numeric_match(
    answer: str,
    hits: list,
    question: str,
    strategy: str,
    tolerance_pct: float = 5.0
) -> Tuple[bool, list, float]:
    """Check if numeric answer matches evidence within tolerance.
    
    Returns:
        (matched: bool, list of matching record_ids, tolerance_pct used)
    """
    try:
        answer_num = float(answer.replace(",", ""))
    except (ValueError, AttributeError):
        return False, [], 0.0
    
    matching_ids = []
    tolerance = answer_num * (tolerance_pct / 100.0)
    
    for hit in hits:
        record = hit.record
        
        try:
            primary_val = str(getattr(record, "primary_value", "")).replace(",", "")
            hit_num = float(primary_val)
            
            if abs(hit_num - answer_num) <= tolerance:
                matching_ids.append(record.record_id)
                continue
        except (ValueError, AttributeError):
            pass
    
    # For quantitative questions, be stricter
    if strategy == "quantitative_calculation" and tolerance_pct > 2.0:
        # Retry with stricter tolerance
        matching_ids_strict = []
        for hit in hits:
            try:
                primary_val = str(getattr(hit.record, "primary_value", "")).replace(",", "")
                hit_num = float(primary_val)
                if abs(hit_num - answer_num) <= (answer_num * 0.02):  # 2% tolerance
                    matching_ids_strict.append(hit.record.record_id)
            except (ValueError, AttributeError):
                pass
        
        if matching_ids_strict:
            return True, matching_ids_strict, 2.0
    
    return len(matching_ids) > 0, matching_ids, tolerance_pct


def _check_partial_support(answer: str, hits: list) -> Tuple[bool, list, str]:
    """Check if answer has partial support from evidence.
    
    Looks for key terms in answer appearing in evidence.
    
    Returns:
        (has_partial_support: bool, list of supporting record_ids, reason: str)
    """
    answer_lower = str(answer).lower()
    
    # Extract key terms (words > 4 chars)
    key_terms = [w for w in answer_lower.split() if len(w) > 4 and not _is_number(w)]
    
    if not key_terms:
        return False, [], ""
    
    supporting_ids = []
    term_matches = {}
    
    for hit in hits:
        record = hit.record
        
        # Check content text for key terms
        content = str(getattr(record, "content_text", "")).lower()
        matched_terms = [t for t in key_terms if t in content]
        
        if matched_terms:
            supporting_ids.append(record.record_id)
            for term in matched_terms:
                term_matches[term] = term_matches.get(term, 0) + 1
    
    if len(supporting_ids) > 0 and len(term_matches) >= len(key_terms) * 0.5:
        matched_term_str = ", ".join(sorted(term_matches.keys())[:3])
        return True, supporting_ids, f"Key terms found: {matched_term_str}"
    
    return False, supporting_ids, ""


def _is_numeric_answer(answer: str) -> bool:
    """Check if answer is primarily numeric."""
    try:
        float(answer.replace(",", "").strip())
        return True
    except (ValueError, AttributeError):
        return bool(re.search(r"[-+]?\d+\.?\d*", str(answer)))


def _is_boolean_answer(answer: str) -> bool:
    text = str(answer).strip().lower()
    return text in {"yes", "no", "true", "false"}


def _is_multi_value_answer(answer: str) -> bool:
    text = str(answer).strip().lower()
    if not text or text == "insufficient_context":
        return False
    if "," in text:
        return True
    if " and " in text:
        return True
    # At least two numeric tokens indicates a list-style answer.
    return len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)) >= 2


def _infer_expected_answer_shape(question: str) -> str:
    q = str(question).lower()
    if re.search(r"\b(is|are|do|does|did|was|were|can|could|should|has|have|had)\b", q):
        return "boolean"
    if any(token in q for token in (
        "two values", "three values", "four values", "in ascending", "in descending",
        "ranked", "ordered", "both values", "all values", "list",
    )):
        return "multi_value"
    return "single"


def _is_number(text: str) -> bool:
    """Check if text is a number."""
    try:
        float(text)
        return True
    except ValueError:
        return False


def _fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Check if two strings are similar using simple fuzzy matching."""
    text1 = str(text1).lower().strip()
    text2 = str(text2).lower().strip()
    
    if not text1 or not text2:
        return False
    
    # Check if one contains the other
    if text1 in text2 or text2 in text1:
        return True
    
    # Levenshtein-like distance check (simplified)
    if len(text1) < 3 or len(text2) < 3:
        return text1 == text2
    
    # Calculate character overlap
    common_chars = sum(1 for c in set(text1) if c in text2)
    similarity = common_chars / max(len(set(text1)), len(set(text2)))
    
    return similarity >= threshold


def _get_retry_recommendation(strategy: str, hits: list) -> str:
    """Get recommendation for retry strategy."""
    if not hits:
        return "Perform expanded retrieval with additional search terms"
    
    if strategy == "quantitative_calculation":
        return "Retry with numeric-focused retrieval and relax unit constraints"
    elif strategy == "relational_comparison":
        return "Retry to find comparative evidence from different sources"
    elif strategy == "multi_table_reasoning":
        return "Retry to ensure evidence from all required tables present"
    elif strategy == "multistep_reasoning":
        return "Retry to collect intermediate values for all steps"
    else:
        return "Retry with reformulated query"


def generate_retry_strategy(
    original_question: str,
    failed_answer: str,
    verification_result: dict,
    strategy: str,
    current_hit_count: int = 0,
) -> dict[str, Any]:
    """Generate strategy for retrying after verification failure.
    
    Returns:
        {
            "retry_type": "expand_query" | "use_next_candidate" | "relax_constraints",
            "new_query": str (if expand_query),
            "skip_record_ids": list (if use_next_candidate),
            "reason": str
        }
    """
    
    # Strategy 1: Try next best candidate if we have enough hits
    if current_hit_count >= 3:
        return {
            "retry_type": "use_next_candidate",
            "skip_record_ids": verification_result.get("used_evidence_ids", []),
            "new_query": None,
            "reason": f"Try next-best candidates among {current_hit_count} retrieved hits"
        }
    
    # Strategy 2: Expand query with additional terms
    expanded_query = _build_expanded_query(original_question, failed_answer, strategy)
    
    return {
        "retry_type": "expand_query",
        "new_query": expanded_query,
        "skip_record_ids": [],
        "reason": f"Expand retrieval: {strategy} failed on initial evidence"
    }


def _build_expanded_query(
    original_question: str,
    failed_answer: str,
    strategy: str,
) -> str:
    """Build expanded query for retry retrieval."""
    components = [original_question]
    
    # Add strategy-specific terms
    if strategy == "quantitative_calculation":
        components.append("numeric values calculation sum total average")
    elif strategy == "relational_comparison":
        components.append("comparative highest lowest greatest comparison")
    elif strategy == "multi_table_reasoning":
        components.append("across tables aggregate multiple sources all facilities")
    elif strategy == "multistep_reasoning":
        components.append("step by step intermediate values stages")
    
    # Add terms from failed answer
    answer_terms = [w for w in str(failed_answer).split() if len(w) > 3]
    if answer_terms:
        components.append(" ".join(answer_terms[:3]))
    
    return " | ".join(components)


def select_fallback_answer(
    original_answer: str,
    alternative_candidates: list,
    verification_results: Optional[list] = None,
) -> Tuple[str, str]:
    """Select best fallback answer from alternatives.
    
    Args:
        original_answer: The initially produced answer
        alternative_candidates: List of alternative answer candidates
        verification_results: Optional verification results for each candidate
    
    Returns:
        (selected_answer: str, source: "initial" | "fallback" | "conservative")
    """
    
    # If original answer has good evidence, keep it
    if original_answer and original_answer != "INSUFFICIENT_CONTEXT":
        return original_answer, "initial"
    
    # Try to find best alternative with evidence
    if verification_results:
        for i, candidate in enumerate(alternative_candidates):
            if i < len(verification_results):
                result = verification_results[i]
                if result.get("verified"):
                    return candidate, "fallback"
    
    # Return first non-empty alternative
    for candidate in alternative_candidates:
        if candidate and candidate != "INSUFFICIENT_CONTEXT":
            return candidate, "fallback"
    
    # Return conservative answer
    return "INSUFFICIENT_CONTEXT", "conservative"
