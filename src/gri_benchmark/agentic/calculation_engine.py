"""Deterministic calculation engine for quantitative questions.

Extracts numeric values from evidence, detects required operations,
and performs calculations with full tracing and citation.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from statistics import mean, median

from gri_benchmark.evidence import RetrievedEvidence


@dataclass
class CalculationResult:
    """Result from deterministic calculation."""
    
    success: bool
    operation: Optional[str]
    computed_result: Optional[str]
    input_values: List[float]
    source_record_ids: List[str]
    units: List[str]
    years: List[str]
    confidence: float
    reason: str
    fallback: bool = False


def extract_numeric_values(
    hits: List[RetrievedEvidence],
    operation: Optional[str] = None,
) -> Tuple[List[float], List[str], List[str], List[str]]:
    """Extract numeric values from retrieved evidence.
    
    For sum/average operations, groups by year to prevent multi-column averaging.
    
    Returns:
        (numeric_values, record_ids, units, years)
    """
    values = []
    record_ids = []
    units_list = []
    years_list = []
    
    # For sum/average: group by year and keep first value per year
    if operation in ["sum", "average"]:
        values_by_year = {}
        for hit in hits:
            record = hit.record
            primary = str(getattr(record, "primary_value", "")).strip()
            
            # Try to extract numeric value
            numeric = _extract_numeric(primary)
            if numeric is not None:
                years = getattr(record, "years", ())
                if years:
                    year = years[0] if isinstance(years, tuple) else years
                    # Keep first value per year
                    if year not in values_by_year:
                        values_by_year[year] = {
                            "value": numeric,
                            "record_id": record.record_id,
                            "units": getattr(record, "units", ()),
                        }
        
        # Extract grouped values
        for year in sorted(values_by_year.keys()):
            entry = values_by_year[year]
            values.append(entry["value"])
            record_ids.append(entry["record_id"])
            units = entry["units"]
            if units:
                units_list.append(units[0] if isinstance(units, tuple) else units)
            years_list.append(year)
    else:
        # Non-grouping path (for other operations)
        for hit in hits:
            record = hit.record
            primary = str(getattr(record, "primary_value", "")).strip()
            
            # Try to extract numeric value
            numeric = _extract_numeric(primary)
            if numeric is not None:
                values.append(numeric)
                record_ids.append(record.record_id)
                
                # Get units and years
                units = getattr(record, "units", ())
                years = getattr(record, "years", ())
                
                if units:
                    units_list.append(units[0] if isinstance(units, tuple) else units)
                if years:
                    years_list.append(years[0] if isinstance(years, tuple) else years)
    
    return values, record_ids, units_list, years_list


def _extract_numeric(value_str: str) -> Optional[float]:
    """Extract numeric value from string, handling commas and percentages."""
    if not value_str:
        return None
    
    try:
        # Remove commas
        value_str = value_str.replace(",", "")
        
        # Handle percentages
        if "%" in value_str:
            value_str = value_str.replace("%", "").strip()
        
        # Convert to float
        return float(value_str)
    except (ValueError, AttributeError):
        return None


def detect_operation(
    question: str,
    values: List[float],
    evidence_hits: List[RetrievedEvidence],
) -> Tuple[Optional[str], float]:
    """Detect required operation from question and evidence.
    
    Returns:
        (operation, confidence)
    
    Operations: sum, average, difference, percentage_change, ratio, max, min, comparison
    """
    if not values:
        return None, 0.0
    
    q_lower = question.lower()
    
    # Average keywords - CHECK BEFORE SUM to avoid false positives
    avg_keywords = ["average", "mean", "typical", "median", "annual"]
    if any(kw in q_lower for kw in avg_keywords):
        return "average", 0.95
    
    # Sum keywords
    sum_keywords = ["sum", "total", "combined", "aggregate", "all together"]
    if any(kw in q_lower for kw in sum_keywords):
        return "sum", 0.95
    
    # Difference keywords
    diff_keywords = ["difference", "difference between", "how much more", "how much less", "reduction", "reduced"]
    if any(kw in q_lower for kw in diff_keywords):
        if len(values) >= 2:
            return "difference", 0.90
    
    # Percentage change keywords
    pct_keywords = [
        "percentage change", "percent increase", "percent decrease",
        "increased by", "decreased by", "change of", "% change"
    ]
    if any(kw in q_lower for kw in pct_keywords):
        if len(values) >= 2:
            return "percentage_change", 0.90
    
    # Ratio keywords
    ratio_keywords = ["ratio", "times", "relative", "compared to"]
    if any(kw in q_lower for kw in ratio_keywords):
        if len(values) >= 2:
            return "ratio", 0.85
    
    # Max/Min keywords
    if "maximum" in q_lower or "highest" in q_lower or "largest" in q_lower:
        if len(values) > 1:
            return "max", 0.90
    
    if "minimum" in q_lower or "lowest" in q_lower or "smallest" in q_lower:
        if len(values) > 1:
            return "min", 0.90
    
    # Comparison keywords
    comparison_keywords = ["compare", "comparison", "which is"]
    if any(kw in q_lower for kw in comparison_keywords):
        if len(values) >= 2:
            return "comparison", 0.80
    
    return None, 0.0


def perform_calculation(
    operation: str,
    values: List[float],
) -> Optional[str]:
    """Perform the detected calculation.
    
    Returns:
        Formatted result string, or None if calculation fails
    """
    if not values:
        return None
    
    try:
        if operation == "sum":
            result = sum(values)
            return _format_result(result)
        
        elif operation == "average":
            result = mean(values)
            return _format_result(result)
        
        elif operation == "median":
            result = median(values)
            return _format_result(result)
        
        elif operation == "difference":
            if len(values) >= 2:
                # Assume first value is baseline
                result = values[1] - values[0]
                return _format_result(result)
        
        elif operation == "percentage_change":
            if len(values) >= 2 and values[0] != 0:
                result = ((values[1] - values[0]) / values[0]) * 100
                return f"{_format_result(result)}%"
        
        elif operation == "ratio":
            if len(values) >= 2 and values[1] != 0:
                result = values[0] / values[1]
                return _format_result(result)
        
        elif operation == "max":
            result = max(values)
            return _format_result(result)
        
        elif operation == "min":
            result = min(values)
            return _format_result(result)
        
        elif operation == "comparison":
            # For comparison, return all values for context
            formatted = [_format_result(v) for v in values]
            return ", ".join(formatted)
    
    except (ValueError, ZeroDivisionError, IndexError):
        return None
    
    return None


def _format_result(value: float) -> str:
    """Format numeric result for display."""
    # Handle very small/large numbers
    if abs(value) >= 1e9:
        return f"{value:.2e}"
    
    # For reasonable values, use appropriate precision
    if value == int(value):
        return str(int(value))
    
    # Round to 2 decimal places, remove trailing zeros
    formatted = f"{value:.2f}".rstrip('0').rstrip('.')
    return formatted


def calculate_for_question(
    question: str,
    retrieved: List[RetrievedEvidence],
    strategy: str,
) -> CalculationResult:
    """Perform calculation for quantitative question.
    
    Args:
        question: The question being asked
        retrieved: Retrieved evidence hits
        strategy: Question strategy (quantitative_calculation or multistep_reasoning)
    
    Returns:
        CalculationResult with computation details
    """
    # Only attempt calculation for quantitative/multistep strategies
    if strategy not in ["quantitative_calculation", "multistep_reasoning"]:
        return CalculationResult(
            success=False,
            operation=None,
            computed_result=None,
            input_values=[],
            source_record_ids=[],
            units=[],
            years=[],
            confidence=0.0,
            reason=f"Strategy '{strategy}' not quantitative",
        )
    
    if not retrieved:
        return CalculationResult(
            success=False,
            operation=None,
            computed_result=None,
            input_values=[],
            source_record_ids=[],
            units=[],
            years=[],
            confidence=0.0,
            reason="No evidence retrieved",
        )
    
    # Detect operation first (needed for smart value extraction)
    # First pass with dummy values to detect operation type
    dummy_values = [1.0] * len(retrieved)
    operation, op_confidence = detect_operation(question, dummy_values, retrieved)
    
    # Extract numeric values with operation awareness for grouping
    values, record_ids, units, years = extract_numeric_values(retrieved, operation=operation)
    
    if not values:
        return CalculationResult(
            success=False,
            operation=None,
            computed_result=None,
            input_values=[],
            source_record_ids=[],
            units=[],
            years=[],
            confidence=0.0,
            reason="No numeric values found in evidence",
        )
    
    # Re-detect operation with actual values for confidence calibration
    operation, op_confidence = detect_operation(question, values, retrieved)
    
    if not operation or op_confidence < 0.75:
        return CalculationResult(
            success=False,
            operation=operation,
            computed_result=None,
            input_values=values,
            source_record_ids=record_ids,
            units=units,
            years=years,
            confidence=op_confidence,
            reason=f"Operation detection confidence too low ({op_confidence:.2f})",
            fallback=True,
        )
    
    # Perform calculation
    result = perform_calculation(operation, values)
    
    if result is None:
        return CalculationResult(
            success=False,
            operation=operation,
            computed_result=None,
            input_values=values,
            source_record_ids=record_ids,
            units=units,
            years=years,
            confidence=op_confidence,
            reason=f"Calculation failed for operation '{operation}'",
            fallback=True,
        )
    
    return CalculationResult(
        success=True,
        operation=operation,
        computed_result=result,
        input_values=values,
        source_record_ids=record_ids,
        units=units,
        years=years,
        confidence=op_confidence,
        reason="Calculation successful",
        fallback=False,
    )


def build_calculation_trace(result: CalculationResult, retrieved: List[RetrievedEvidence]) -> Dict[str, Any]:
    """Build detailed calculation trace for metadata.
    
    Returns:
        Dict with calculation details for storage in prediction.metadata
    """
    # Map record IDs to citations
    source_evidence = []
    record_id_to_hit = {hit.record.record_id: hit for hit in retrieved}
    
    for rec_id in result.source_record_ids:
        if rec_id in record_id_to_hit:
            hit = record_id_to_hit[rec_id]
            record = hit.record
            source_evidence.append({
                "record_id": rec_id,
                "value": str(getattr(record, "primary_value", "")),
                "source_file": getattr(record, "source_file", ""),
                "table_id": getattr(record, "table_id", ""),
                "row_id": getattr(record, "row_id", ""),
                "column_id": getattr(record, "column_id", ""),
            })
    
    trace = {
        "operation": result.operation,
        "input_values": result.input_values,
        "units": result.units,
        "years": result.years,
        "computed_result": result.computed_result,
        "source_evidence": source_evidence,
        "confidence": result.confidence,
        "success": result.success,
        "reason": result.reason,
    }
    
    if result.fallback:
        trace["fallback"] = True
    
    return trace
