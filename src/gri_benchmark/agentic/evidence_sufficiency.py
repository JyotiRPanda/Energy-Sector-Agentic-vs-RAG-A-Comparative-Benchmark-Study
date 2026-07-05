"""Evidence sufficiency checking for agentic multi-tool pipeline.

Evaluates whether retrieved evidence is sufficient for answering a question
based on question type and expected answer characteristics.
"""

import re
from typing import Any, Optional


def evidence_sufficiency_check(
    question: str,
    retrieved_hits: list,
    strategy: str,
) -> dict[str, Any]:
    """Check if retrieved evidence is sufficient for answering question.
    
    Args:
        question: The user question
        retrieved_hits: List of RetrievedEvidence objects
        strategy: Question routing strategy (extractive_lookup, relational_comparison, 
                 quantitative_calculation, multistep_reasoning, multi_table_reasoning)
    
    Returns:
        {
            "sufficient": bool,
            "confidence": float between 0 and 1,
            "missing": list of missing elements,
            "required_fields": list of required fields for this strategy,
            "reason": string explanation
        }
    """
    if not retrieved_hits:
        return {
            "sufficient": False,
            "confidence": 0.0,
            "missing": ["any evidence"],
            "required_fields": _get_required_fields_for_strategy(strategy),
            "reason": "No evidence retrieved"
        }
    
    # Delegate to strategy-specific checker
    if strategy == "extractive_lookup":
        return _check_extractive_sufficiency(question, retrieved_hits)
    elif strategy == "relational_comparison":
        return _check_relational_sufficiency(question, retrieved_hits)
    elif strategy == "quantitative_calculation":
        return _check_quantitative_sufficiency(question, retrieved_hits)
    elif strategy == "multistep_reasoning":
        return _check_multistep_sufficiency(question, retrieved_hits)
    elif strategy == "multi_table_reasoning":
        return _check_multi_table_sufficiency(question, retrieved_hits)
    else:
        # Default to extractive for unknown
        return _check_extractive_sufficiency(question, retrieved_hits)


def _get_required_fields_for_strategy(strategy: str) -> list[str]:
    """Get list of required fields for each strategy."""
    base_fields = ["primary_value", "content_text"]
    
    strategy_fields = {
        "extractive_lookup": base_fields + ["source_file", "table_id"],
        "relational_comparison": base_fields + ["source_file", "table_id", "table_id"],  # 2+ tables
        "quantitative_calculation": base_fields + ["years", "units"],
        "multistep_reasoning": base_fields + ["years", "units", "intents"],
        "multi_table_reasoning": base_fields + ["source_file", "table_id", "years", "units"],
    }
    
    return strategy_fields.get(strategy, base_fields)


def _check_extractive_sufficiency(question: str, hits: list) -> dict[str, Any]:
    """Check sufficiency for simple extractive questions.
    
    Requires:
    - At least one hit with a plausible answer value
    - Relevant source/table metadata present
    """
    required = ["primary_value", "content_text", "source_file", "table_id"]
    missing = []
    confidence_scores = []
    
    for hit in hits:
        hit_missing = []
        hit_confidence = 1.0
        
        # Check required fields
        for field in required:
            value = getattr(hit.record, field, None)
            if not value:
                hit_missing.append(field)
                hit_confidence -= 0.15
        
        if not hit_missing:
            # This hit has all required fields
            confidence_scores.append(hit_confidence)
    
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        return {
            "sufficient": True,
            "confidence": min(avg_confidence, 1.0),
            "missing": [],
            "required_fields": required,
            "reason": f"At least one hit contains plausible answer value with metadata"
        }
    
    # Check which fields are missing across all hits
    all_missing_fields = set()
    for hit in hits:
        for field in required:
            if not getattr(hit.record, field, None):
                all_missing_fields.add(field)
    
    confidence = 1.0 - (0.25 * len(all_missing_fields))
    confidence = max(0.0, confidence)
    
    return {
        "sufficient": False,
        "confidence": confidence,
        "missing": list(all_missing_fields),
        "required_fields": required,
        "reason": f"Missing required fields: {', '.join(all_missing_fields)}"
    }


def _check_relational_sufficiency(question: str, hits: list) -> dict[str, Any]:
    """Check sufficiency for relational/comparison questions.
    
    Requires:
    - At least two comparable candidate values
    - From same or different sources (for comparison)
    """
    required = ["primary_value", "content_text", "source_file", "table_id"]
    
    # Extract candidate values
    candidates = []
    for hit in hits:
        value = getattr(hit.record, "primary_value", None)
        if value:
            candidates.append({
                "value": value,
                "source": getattr(hit.record, "source_file", "unknown"),
                "table": getattr(hit.record, "table_id", "unknown"),
                "record": hit.record
            })
    
    # Check for metadata
    missing = []
    for field in required:
        if not any(getattr(hit.record, field, None) for hit in hits):
            missing.append(field)
    
    if len(candidates) >= 2:
        # Have multiple candidates for comparison
        confidence = min(0.95, 0.70 + (0.15 * min(len(candidates), 5)))
        return {
            "sufficient": True,
            "confidence": confidence,
            "missing": missing,
            "required_fields": required,
            "reason": f"Found {len(candidates)} comparable candidates for relational query"
        }
    elif len(candidates) == 1:
        # Only one candidate - insufficient for comparison
        return {
            "sufficient": False,
            "confidence": 0.3,
            "missing": ["comparative_candidate"],
            "required_fields": required,
            "reason": "Need at least 2 candidates for comparison, found only 1"
        }
    else:
        return {
            "sufficient": False,
            "confidence": 0.0,
            "missing": ["any_primary_value"] + missing,
            "required_fields": required,
            "reason": "No candidate values found for comparison"
        }


def _check_quantitative_sufficiency(question: str, hits: list) -> dict[str, Any]:
    """Check sufficiency for quantitative questions.
    
    Requires:
    - Numeric values for calculation
    - Years/time references if asking about temporal data
    - Units matching or compatible with question
    """
    required = ["primary_value", "content_text", "years", "units"]
    missing = []
    confidence_scores = []
    
    # Extract years and units from question
    question_lower = question.lower()
    years_mentioned = re.findall(r"\b(20\d{2}|19\d{2})\b", question_lower)
    units_mentioned = _extract_units_from_question(question_lower)
    
    has_numeric = False
    has_year_match = False
    has_unit_match = False
    
    for hit in hits:
        record = hit.record
        
        # Check for numeric value
        primary_val = getattr(record, "primary_value", None)
        if primary_val and _is_numeric(str(primary_val)):
            has_numeric = True
        
        # Check year match
        hit_years = getattr(record, "years", ())
        if years_mentioned and any(y in hit_years for y in years_mentioned):
            has_year_match = True
        
        # Check unit match
        hit_units = getattr(record, "units", ())
        if units_mentioned and any(u in hit_units for u in units_mentioned):
            has_unit_match = True
    
    # Calculate confidence
    confidence = 0.0
    hit_missing = []
    
    if not has_numeric:
        hit_missing.append("numeric_value")
    else:
        confidence += 0.4
    
    if years_mentioned and not has_year_match:
        hit_missing.append(f"matching_years_{years_mentioned}")
    elif years_mentioned:
        confidence += 0.3
    else:
        confidence += 0.15  # No year requirement but still good
    
    if units_mentioned and not has_unit_match:
        hit_missing.append(f"matching_units_{units_mentioned}")
    elif units_mentioned:
        confidence += 0.3
    else:
        confidence += 0.15  # No unit requirement but still good
    
    sufficient = has_numeric and (not years_mentioned or has_year_match)
    
    return {
        "sufficient": sufficient,
        "confidence": min(confidence, 1.0),
        "missing": hit_missing,
        "required_fields": required,
        "reason": (
            f"Numeric: {has_numeric}, Years match: {has_year_match}, Units match: {has_unit_match}"
        )
    }


def _check_multistep_sufficiency(question: str, hits: list) -> dict[str, Any]:
    """Check sufficiency for multi-step reasoning questions.
    
    Requires:
    - Multiple numeric values or calculation steps
    - Consistent years and units across steps
    - Intermediate values for stage operations
    """
    required = ["primary_value", "years", "units", "intents"]
    
    # Extract operation types from question
    operations = _extract_operations_from_question(question)
    
    numeric_hits = [
        hit for hit in hits 
        if _is_numeric(str(getattr(hit.record, "primary_value", None)))
    ]
    
    missing = []
    confidence = 0.0
    
    if not numeric_hits:
        missing.append("numeric_values_for_operations")
        return {
            "sufficient": False,
            "confidence": 0.0,
            "missing": missing,
            "required_fields": required,
            "reason": "No numeric values found for multi-step operations"
        }
    
    # Check for variety in values (different steps)
    unique_intents = set()
    for hit in numeric_hits:
        intents = getattr(hit.record, "intents", ())
        unique_intents.update(intents)
    
    if len(numeric_hits) >= 2:
        confidence = 0.7 + min(0.2, len(unique_intents) * 0.1)
        return {
            "sufficient": True,
            "confidence": confidence,
            "missing": [],
            "required_fields": required,
            "reason": f"Found {len(numeric_hits)} numeric values for multi-step reasoning"
        }
    else:
        confidence = 0.4
        missing.append("additional_intermediate_values")
        return {
            "sufficient": False,
            "confidence": confidence,
            "missing": missing,
            "required_fields": required,
            "reason": "Insufficient numeric values for multi-step reasoning"
        }


def _check_multi_table_sufficiency(question: str, hits: list) -> dict[str, Any]:
    """Check sufficiency for multi-table reasoning questions.
    
    Requires:
    - Evidence from multiple source tables
    - Consistent years/units across tables
    - Non-conflicting metadata
    """
    required = ["source_file", "table_id", "years", "units"]
    
    # Extract unique table references
    tables = set()
    sources = set()
    years_set = set()
    units_set = set()
    
    for hit in hits:
        table_id = getattr(hit.record, "table_id", None)
        source = getattr(hit.record, "source_file", None)
        
        if table_id:
            tables.add(str(table_id))
        if source:
            sources.add(str(source))
        
        years = getattr(hit.record, "years", ())
        units = getattr(hit.record, "units", ())
        years_set.update(years)
        units_set.update(units)
    
    missing = []
    
    if len(tables) < 2:
        missing.append("evidence_from_second_table")
    if not sources:
        missing.append("source_file_metadata")
    
    confidence = 0.0
    if len(tables) >= 2:
        confidence += 0.6
    elif len(tables) == 1:
        confidence += 0.3
    
    if len(sources) > 0:
        confidence += 0.2
    
    if years_set:
        confidence += 0.1
    if units_set:
        confidence += 0.1
    
    sufficient = len(tables) >= 2 and len(sources) > 0
    
    return {
        "sufficient": sufficient,
        "confidence": min(confidence, 1.0),
        "missing": missing,
        "required_fields": required,
        "reason": f"Found {len(tables)} table(s) and {len(sources)} source(s)"
    }


def _is_numeric(value: str) -> bool:
    """Check if value contains numeric content."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return bool(re.search(r"[-+]?\d*\.?\d+", str(value)))


def _extract_units_from_question(question_lower: str) -> list[str]:
    """Extract likely units from question text."""
    units = []
    
    unit_patterns = {
        "mwh": ["mwh", "megawatt-hour", "megawatt hour"],
        "gwh": ["gwh", "gigawatt-hour", "gigawatt hour"],
        "kwh": ["kwh", "kilowatt-hour", "kilowatt hour"],
        "tonnes": ["tonnes", "tons", "metric tons", "tonne", "ton"],
        "kg": ["kg", "kilogram", "kilograms"],
        "liters": ["liter", "liters", "litre", "litres"],
        "m3": ["m3", "cubic meter", "cubic metres"],
        "percent": ["percent", "%", "percentage"],
    }
    
    for unit, patterns in unit_patterns.items():
        for pattern in patterns:
            if pattern in question_lower:
                units.append(unit)
                break
    
    return units


def _extract_operations_from_question(question_lower: str) -> list[str]:
    """Extract calculation operations from question."""
    operations = []
    
    operation_patterns = {
        "sum": ["sum", "total", "aggregate", "combined"],
        "average": ["average", "mean", "median"],
        "difference": ["difference", "change", "delta"],
        "ratio": ["ratio", "proportion", "fraction"],
        "percent": ["percent", "percentage", "percentage change"],
    }
    
    for op, patterns in operation_patterns.items():
        for pattern in patterns:
            if pattern in question_lower:
                operations.append(op)
                break
    
    return operations


def build_retry_query(
    original_question: str,
    retrieved_hits: list,
    detected_metadata: Optional[dict] = None
) -> str:
    """Build expanded query for retry retrieval after insufficient evidence.
    
    Args:
        original_question: Original user question
        retrieved_hits: Previously retrieved hits (to avoid)
        detected_metadata: Detected years, units, GRI codes, etc.
    
    Returns:
        Expanded query string with additional search signals
    """
    if detected_metadata is None:
        detected_metadata = {}
    
    components = [original_question]
    
    # Add detected temporal references
    years = detected_metadata.get("years", [])
    if years:
        components.append(f"years: {' '.join(years)}")
    
    # Add detected units
    units = detected_metadata.get("units", [])
    if units:
        components.append(f"units: {' '.join(units)}")
    
    # Add GRI code if available
    gri_code = detected_metadata.get("gri_code")
    if gri_code:
        components.append(f"GRI {gri_code}")
    
    # Add source identifiers
    sources = detected_metadata.get("sources", [])
    if sources:
        components.append(f"sources: {' '.join(sources)}")
    
    return " | ".join(components)
