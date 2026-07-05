"""
Table-Aware Grounding Layer for GRI-QA Agentic Pipeline

Key insight: GRI-QA dataset is built on table structure (row_id, col_id relationships).
Operands must be validated using table structure, not text extraction alone.

This grounding layer:
1. Filters for table-based operands only (is_table_data=True)
2. Validates operand coherence via table structure (same table_id, consistent row/col relationships)
3. Uses pre-annotated metadata (intents, years, units)
4. Reconstructs operation intent from table structure context
5. Refuses to process if table structure is inconsistent
"""

from dataclasses import dataclass
from typing import Optional
from collections import defaultdict, Counter
import re
import sys

from gri_benchmark.evidence import RetrievedEvidence


@dataclass(frozen=True)
class TableAwareCandidate:
    """A candidate number validated through table structure."""
    value: float
    source_chunk: RetrievedEvidence
    source_text: str
    
    # Table structure info
    table_id: Optional[str]
    row_id: Optional[str]
    column_id: Optional[str]
    years: tuple[str, ...]
    units: tuple[str, ...]
    annotated_intents: tuple[str, ...]
    
    # Validation
    confidence: float  # Based on table coherence, not text relevance
    coherence_reason: str  # Why this operand is valid


def extract_question_years(question: str) -> set[str]:
    """Extract years mentioned in the question."""
    # Fixed regex: capture full year, not just the prefix
    years = set(re.findall(r"(?:19|20)\d{2}", question))
    return years


def extract_question_metric_type(question: str) -> Optional[str]:
    """Extract primary metric type from question text.
    
    Returns: 'waste', 'energy', 'emissions', 'water', 'other', or None
    """
    question_lower = question.lower()
    
    # Order matters - check more specific terms first
    metric_keywords = {
        'waste': ['waste', 'refuse', 'scrap', 'byproduct'],
        'energy': ['energy', 'electricity', 'fuel', 'power', 'kwh', 'mwh', 'gj', 'joule'],
        'emissions': ['emissions', 'ghg', 'carbon', 'co2', 'co₂', 'greenhouse', 'methane', 'scope'],
        'water': ['water', 'discharge', 'wastewater', 'effluent', 'm³', 'cubic'],
    }
    
    # Match keywords in order of specificity
    for metric_type, keywords in metric_keywords.items():
        for keyword in keywords:
            if keyword in question_lower:
                return metric_type
    
    return None


def extract_table_candidates(
    question: str,
    retrieved_chunks: list[RetrievedEvidence],
) -> list[TableAwareCandidate]:
    """
    Extract candidates from table cells only (is_table_data=True).
    
    Returns candidates with table structure validation.
    Refuses to use metadata/year values from text chunks.
    """
    candidates: list[TableAwareCandidate] = []
    
    for chunk in retrieved_chunks:
        rec = chunk.record
        
        # FILTER 1: Must have table identity (table_id/row_id/column_id)
        # Note: is_table_data defaults to False, so we check structure directly
        if not rec.table_id or not rec.row_id or not rec.column_id:
            continue
        
        # FILTER 2: Must have a numeric value
            continue
        
        # FILTER 3: Extract primary value if available
        if not rec.primary_value:
            continue
        
        try:
            value = float(rec.primary_value)
        except ValueError:
            continue
        
        # Check if value makes sense (not a year, not obviously metadata)
        if 2000 <= value <= 2030:  # Skip year values
            continue
        
        candidate = TableAwareCandidate(
            value=value,
            source_chunk=chunk,
            source_text=rec.content_text[:100],
            table_id=rec.table_id,
            row_id=rec.row_id,
            column_id=rec.column_id,
            years=rec.years,
            units=rec.units,
            annotated_intents=rec.intents,
            confidence=0.8,  # Default - refined by coherence checks
            coherence_reason="table_cell_with_metadata",
        )
        candidates.append(candidate)
    
    return candidates


def infer_operation_intent(candidates: list[TableAwareCandidate]) -> str:
    """Infer operation from table structure only.
    
    Note: Pre-annotated intents in corpus are not cell-specific - they contain
    ALL intents from the question set. So we rely on inference instead.
    """
    if not candidates:
        return "extractive"
    
    # Inference from table structure
    if len(candidates) == 1:
        return "extractive"
    
    # Check if all from same table but different rows → SUM/AVG
    table_ids = {c.table_id for c in candidates}
    row_ids = {c.row_id for c in candidates}
    
    if len(table_ids) == 1 and len(row_ids) > 1:
        # Multiple rows, same table → likely sum/average
        return "sum"
    
    # Check if same row, different temporal columns
    if len(row_ids) == 1 and len(candidates) >= 2:
        # Same row, multiple candidates → likely temporal difference
        years_list = [c.years for c in candidates]
        if years_list[0] != years_list[-1]:
            return "difference"
    
    # Check if units suggest percentage
    units_set = set()
    for c in candidates:
        units_set.update(c.units)
    
    if any("%" in u or "percent" in u.lower() for u in units_set):
        return "percentage"
    
    # Default
    return "sum" if len(candidates) > 1 else "extractive"


def infer_operation_intent_from_question(question: str) -> str:
    """Extract operation intent directly from question text.
    
    This is more reliable than cell annotations since corpus metadata
    isn't cell-specific.
    """
    question_lower = question.lower()
    
    # Check in order of SPECIFICITY (more specific first)
    
    # Average/mean operations - MORE SPECIFIC (explicit average)
    if any(word in question_lower for word in ['average', 'mean', 'median']):
        return "average"
    
    # Percentage operations - check before generic "reduction"
    if any(word in question_lower for word in ['percent', 'percentage', 'ratio', 'proportion']):
        return "percentage"
    
    # Difference/change operations - explicit keywords
    difference_keywords = ['difference', 'increase', 'decrease', 'change', 'growth', 'decline', 'reduction']
    if any(word in question_lower for word in difference_keywords):
        return "difference"
    
    # Temporal difference - "from X to Y" pattern with years
    temporal_keywords = ['from', 'to', 'between']
    has_temporal = any(f' {word} ' in f' {question_lower} ' for word in temporal_keywords)
    if has_temporal and any(str(year) in question_lower for year in range(2000, 2030)):
        return "difference"
    
    # Sum operations - LESS SPECIFIC (generic total/sum)
    if any(word in question_lower for word in ['sum', 'total', 'combined', 'aggregate']):
        return "sum"
    
    # Extractive (no operation) - default
    return "extractive"

def validate_operand_coherence(
    candidates: list[TableAwareCandidate],
    operation: str,
) -> tuple[list[TableAwareCandidate], float]:
    """
    Validate that operands form coherent group for the given operation.
    
    Returns (validated_operands, coherence_confidence)
    """
    if not candidates:
        return [], 0.0
    
    if operation == "extractive":
        # Single operand - just validate it's from table
        return candidates[:1], 0.9
    
    elif operation in ("sum", "average"):
        # All operands must be from same table
        table_ids = {c.table_id for c in candidates}
        
        if len(table_ids) != 1:
            # Operands from different tables - incoherent
            return [], 0.0
        
        # All units should be consistent
        units_sets = [set(c.units) for c in candidates]
        if units_sets and not all(u == units_sets[0] for u in units_sets):
            # Inconsistent units - lower confidence but still usable
            return candidates, 0.6
        
        return candidates, 0.85
    
    elif operation == "difference":
        # Should be 2 operands from same row, different temporal columns
        # OR 2 operands with different year values (temporal comparison)
        # OR 1 operand when data only has 1 year (fallback)
        if len(candidates) == 0:
            return [], 0.0
        
        if len(candidates) == 1:
            # Single operand - only when data doesn't have multiple years
            # Return with lower confidence
            return candidates, 0.5
        
        candidates = candidates[:2]
        
        # More lenient: either same row OR different years (temporal comparison)
        years_list = [c.years for c in candidates]
        row_ids = {c.row_id for c in candidates}
        
        # Check temporal ordering
        if len(years_list) >= 2:
            has_temporal_diff = years_list[0] and years_list[1] and years_list[0] != years_list[1]
            if has_temporal_diff:
                # Different years - good for difference (even if different rows)
                return candidates, 0.85
            elif len(row_ids) == 1:
                # Same row, different columns (likely temporal)
                return candidates, 0.80
        
        return candidates, 0.70
    
    elif operation == "percentage":
        # Should be 2 operands: part and whole or base and comparison
        # But accept 1 operand when data only has 1 year (temporal percentage change)
        if len(candidates) == 0:
            return [], 0.0
        
        if len(candidates) == 1:
            # Single operand - only when data doesn't have multiple years
            return candidates, 0.5
        
        candidates = candidates[:2]
        
        # Check if units indicate percentage
        units_sets = [set(c.units) for c in candidates]
        has_percent = any("%" in u or "percent" in u.lower() for u_set in units_sets for u in u_set)
        
        if has_percent:
            return candidates, 0.8
        
        # Check temporal difference (for "percentage reduction from year1 to year2")
        years_list = [c.years for c in candidates]
        if len(years_list) >= 2:
            has_temporal_diff = years_list[0] and years_list[1] and years_list[0] != years_list[1]
            if has_temporal_diff:
                return candidates, 0.75
        
        return candidates, 0.6
    
    else:
        # Unknown operation
        return candidates[:1], 0.5


def select_operands_table_aware(
    question: str,
    candidates: list[TableAwareCandidate],
) -> tuple[list[TableAwareCandidate], float, str]:
    """
    Select operands using YEAR-AWARE and METRIC-AWARE matching from row-joined retrieval.
    
    KEY: join-based retrieval stores per-cell year info in cell_map.
    Use cell_map to find operands matching question years AND metric type.
    """
    if not candidates:
        return [], 0.0, "insufficient"
    
    # Step 0: Extract question years and metric type
    question_years = extract_question_years(question)
    question_metric = extract_question_metric_type(question)
    
    # LOGGING: Track filtering process
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"Q: {question[:80]}", file=sys.stderr)
    print(f"Metric: {question_metric}, Years: {question_years}", file=sys.stderr)
    print(f"Candidates BEFORE filtering: {len(candidates)}", file=sys.stderr)
    
    # Step 1: Determine operation intent from question
    operation = infer_operation_intent_from_question(question)
    
    # Step 2: Check if any candidates have row_join_cell_map (from row joining)
    has_row_joined = any(
        "row_join_cell_map" in c.source_chunk.score_breakdown
        for c in candidates
    )
    
    if has_row_joined and operation in ("difference", "average", "sum"):
        # Try to match operands using per-cell year info from row_join_cell_map
        # AND metric type if available
        year_matched_operands = _select_by_year_from_cell_map(
            candidates, question_years, operation, question_metric
        )
        if year_matched_operands:
            validated, conf = validate_operand_coherence(year_matched_operands, operation)
            return validated, conf, operation
    
    # Fall back to standard selection
    if operation == "extractive":
        validated, conf = validate_operand_coherence(candidates[:1], operation)
        return validated, conf, operation
    
    elif operation in ("sum", "average"):
        # FIXED: Use row_join_cell_map to get operands from same row
        # Extract cells from row_join_cell_map for candidates that have it
        cell_candidates = []
        
        for cand in candidates:
            cell_map = cand.source_chunk.score_breakdown.get("row_join_cell_map", {})
            if cell_map:
                # Found row-joined candidate - extract individual cells
                for col_id, cell_data in cell_map.items():
                    year_values = cell_data.get("values_by_year", {})
                    for year, value in year_values.items():
                        # Create a virtual cell candidate
                        cell_cand = TableAwareCandidate(
                            value=value,
                            source_chunk=cand.source_chunk,
                            source_text=cand.source_text,
                            table_id=cand.table_id,
                            row_id=cand.row_id,
                            column_id=col_id,
                            years=(year,),
                            units=cand.units,
                            annotated_intents=cand.annotated_intents,
                            confidence=cand.confidence,
                            coherence_reason="row_join_cell",
                        )
                        cell_candidates.append(cell_cand)
            else:
                # No row-join, use candidate as-is
                cell_candidates.append(cand)
        
        if not cell_candidates:
            return [], 0.0, "no_candidates"
        
        # Group by (table_id, row_id) and track retrieval rank
        groups = {}
        group_first_rank = {}
        for rank, c in enumerate(cell_candidates):
            key = (c.table_id, c.row_id)
            if key not in groups:
                groups[key] = []
                group_first_rank[key] = rank
            groups[key].append(c)
        
        if not groups:
            return [], 0.0, "no_groups"
        
        # Pick best group by RETRIEVAL RANK (not size)
        best_key = min(group_first_rank.keys(), key=lambda k: group_first_rank[k])
        best_group = groups[best_key]
        
        print(f"Operation: {operation}, best_group selected by rank", file=sys.stderr)
        print(f"Groups found: {len(groups)}, best group size: {len(best_group)}", file=sys.stderr)
        
        # FIX: For sum/average, select exactly ONE cell per year (not all cells per year)
        # Group operands by year and keep first (preferred column)
        operands_by_year = {}
        
        # First pass: cells that match question years
        for c in best_group:
            if c.years and any(y in question_years for y in c.years):
                year = c.years[0]  # year is stored as tuple
                if year not in operands_by_year:
                    operands_by_year[year] = c
        
        # If we found cells matching question years, use those (one per year)
        if operands_by_year:
            operands = list(operands_by_year.values())
        else:
            # Fallback: use all cells from best group
            operands = best_group
        
        # HARD CONSTRAINT: reject cross-row pairs
        unique_rows = {c.row_id for c in operands}
        if len(unique_rows) > 1:
            print(f"   REJECTED: Cross-row pair (rows: {unique_rows})", file=sys.stderr)
            return [], 0.0, "invalid_cross_row"
        
        print(f"  ✓ Valid same-row operands: {len(operands)}, columns: {[c.column_id for c in operands]}, years: {[c.years for c in operands]}", file=sys.stderr)
        validated, conf = validate_operand_coherence(operands, operation)
        return validated, conf, operation
    
    elif operation == "difference":
        # FIXED: Use row_join_cell_map to get operands from same row
        # Extract cells from row_join_cell_map for candidates that have it
        cell_candidates = []
        
        for cand in candidates:
            cell_map = cand.source_chunk.score_breakdown.get("row_join_cell_map", {})
            if cell_map:
                # Found row-joined candidate - extract individual cells
                for col_id, cell_data in cell_map.items():
                    year_values = cell_data.get("values_by_year", {})
                    for year, value in year_values.items():
                        # Create a virtual cell candidate
                        cell_cand = TableAwareCandidate(
                            value=value,
                            source_chunk=cand.source_chunk,
                            source_text=cand.source_text,
                            table_id=cand.table_id,
                            row_id=cand.row_id,
                            column_id=col_id,
                            years=(year,),
                            units=cand.units,
                            annotated_intents=cand.annotated_intents,
                            confidence=cand.confidence,
                            coherence_reason="row_join_cell",
                        )
                        cell_candidates.append(cell_cand)
            else:
                # No row-join, use candidate as-is
                cell_candidates.append(cand)
        
        if not cell_candidates:
            return [], 0.0, "no_candidates"
        
        # Group by (table_id, row_id) and track retrieval rank
        groups = {}
        group_first_rank = {}
        for rank, c in enumerate(cell_candidates):
            key = (c.table_id, c.row_id)
            if key not in groups:
                groups[key] = []
                group_first_rank[key] = rank
            groups[key].append(c)
        
        if not groups:
            return [], 0.0, "no_groups"
        
        # Try each group in rank order, pick the first one that has enough years/columns
        sorted_group_keys = sorted(groups.keys(), key=lambda k: group_first_rank[k])
        selected = []
        
        for group_key in sorted_group_keys:
            best_group = groups[group_key]
            
            # Filter by question years within the group
            operands = []
            for c in best_group:
                if c.years and any(y in question_years for y in c.years):
                    operands.append(c)
            
            # Fallback if none matched years
            if len(operands) == 0:
                operands = best_group
            
            # HARD CONSTRAINT: reject cross-row pairs
            unique_rows = {c.row_id for c in operands}
            if len(unique_rows) > 1:
                print(f"   REJECTED: Cross-row pair (rows: {unique_rows})", file=sys.stderr)
                continue  # Try next group
            
            # For difference: check if we have 2+ operands with different columns
            # OR 2+ operands with different years (same column)
            unique_cols = {c.column_id for c in operands}
            unique_years = {c.years[0] if c.years else None for c in operands if c.years}
            
            has_multi_cols = len(unique_cols) >= 2
            has_multi_years = len(unique_years) >= 2
            
            if has_multi_cols or has_multi_years:
                # Found a group with either multiple columns OR multiple years - use it
                if has_multi_years and not has_multi_cols:
                    # Same column, different years - pick first 2 with different years
                    seen_years = set()
                    for c in operands:
                        if c.years and c.years[0] not in seen_years:
                            selected.append(c)
                            seen_years.add(c.years[0])
                            if len(selected) >= 2:
                                break
                    print(f"  ✓ Found group {group_key} with 2 operands from same column, different years", file=sys.stderr)
                else:
                    # Different columns - pick first 2 with different columns
                    seen_cols = set()
                    for c in operands:
                        if c.column_id not in seen_cols:
                            selected.append(c)
                            seen_cols.add(c.column_id)
                            if len(selected) >= 2:
                                break
                    print(f"  ✓ Found group {group_key} with {len(selected)} operands from {len(unique_cols)} columns", file=sys.stderr)
                break
            else:
                # Only 1 column and 1 year - keep for fallback
                print(f"  ℹ️  Group {group_key} has only 1 column and 1 year, checking next groups", file=sys.stderr)
                continue  # Try next group first
        
        # If no group with 2+ columns/years found, accept single best operand
        if not selected and groups:
            # Fallback: take first operand from first group
            best_group = groups[sorted_group_keys[0]]
            operands = []
            for c in best_group:
                if c.years and any(y in question_years for y in c.years):
                    operands.append(c)
            if not operands:
                operands = best_group
            
            print(f"  ℹ️  Fallback: accepting single operand from best group", file=sys.stderr)
            selected = [operands[0]]
        
        if len(selected) < 2 and len(operands) >= 2:
            selected = operands[:2]
        
        print(f"  ✓ Valid same-row operands: {len(selected)}, cols: {[c.column_id for c in selected]}", file=sys.stderr)
        validated, conf = validate_operand_coherence(selected, operation)
        return validated, conf, operation
    

    # Default fallback for other operations
    validated, conf = validate_operand_coherence(candidates, operation)
    return validated, conf, operation

def _select_by_year_from_cell_map(
    candidates: list[TableAwareCandidate],
    question_years: set[str],
    operation: str,
    question_metric: Optional[str] = None,
) -> list[TableAwareCandidate]:
    """
    Try to match operands using per-cell year info from row_join_cell_map.
    
    Also filters by metric type if question_metric is provided.
    
    Returns operands matched to question years AND metric type, or empty list if can't match.
    """
    if not question_years:
        return []
    
    # Collect all cells with their year->value mappings from cell_map
    cells_with_years = []
    for cand in candidates:
        # Filter by metric type if specified
        if question_metric:
            candidate_metric = cand.source_chunk.score_breakdown.get("metric_type")
            if candidate_metric and candidate_metric != question_metric and candidate_metric != "unknown":
                continue  # Skip candidates with different metric type
        
        cell_map = cand.source_chunk.score_breakdown.get("row_join_cell_map", {})
        for col_id, cell_data in cell_map.items():
            year_values = cell_data.get("values_by_year", {})
            if year_values:
                for year, value in year_values.items():
                    cells_with_years.append({
                        "col_id": col_id,
                        "year": year,
                        "value": value,
                        "candidate": cand,
                    })
    
    if not cells_with_years:
        return []
    
    if operation == "difference" and len(question_years) >= 2:
        # Match cells to each question year
        question_years_list = sorted(list(question_years))
        matched = []
        for year in question_years_list:
            for cell in cells_with_years:
                if cell["year"] == year and cell not in matched:
                    matched.append(cell)
                    break
        
        if len(matched) >= 2:
            # Return unique candidates (may be same candidate with different years)
            return [cell["candidate"] for cell in matched]
    
    elif operation in ("sum", "average"):
        # FIX: For sum/average, select exactly ONE cell per year (not all cells per year)
        # This prevents averaging multiple columns for the same year
        matched_by_year = {}  # year -> cell
        for cell in cells_with_years:
            if cell["year"] in question_years:
                # Keep first cell found for each year (prefer first column)
                if cell["year"] not in matched_by_year:
                    matched_by_year[cell["year"]] = cell
        
        if matched_by_year:
            # Return cells (wrapped in candidates for API compat)
            # For sum/average we need to return individual cells, not candidates
            # Create synthetic candidates for each cell
            result_candidates = []
            for year in sorted(matched_by_year.keys()):
                cell = matched_by_year[year]
                # Create a synthetic candidate with just this cell's value
                # Use the original candidate as a template
                orig_cand = cell["candidate"]
                cell_cand = TableAwareCandidate(
                    value=cell["value"],
                    source_chunk=orig_cand.source_chunk,
                    source_text=orig_cand.source_text,
                    table_id=orig_cand.table_id,
                    row_id=orig_cand.row_id,
                    column_id=cell["col_id"],
                    years=(cell["year"],),
                    units=orig_cand.units,
                    annotated_intents=orig_cand.annotated_intents,
                    confidence=orig_cand.confidence,
                    coherence_reason="cell_from_row_join",
                )
                result_candidates.append(cell_cand)
            return result_candidates
    
    return []


def should_use_table_grounded_operands(
    operands: list[TableAwareCandidate],
    confidence: float,
    threshold: float = 0.6,
) -> bool:
    """
    Determine if operands are confident enough to use for calculation.
    
    Conservative approach: Refuse to process if confidence is below threshold.
    """
    if not operands:
        return False
    
    if confidence < threshold:
        return False
    
    # All operands must have complete table identity
    if not all(c.row_id and c.column_id and c.table_id for c in operands):
        return False
    
    return True


def calculate_from_table_operands(
    operands: list[TableAwareCandidate],
    operation: str,
) -> tuple[str, str]:
    """
    Perform calculation on table-validated operands.
    
    Returns (answer, trace)
    """
    if not operands:
        return "INSUFFICIENT_CONTEXT", "No operands available"
    
    values = [c.value for c in operands]
    
    if operation == "extractive":
        # Return single value
        answer = str(int(values[0]) if values[0] == int(values[0]) else values[0])
        trace = f"Extracted value from table cell [{operands[0].row_id}, {operands[0].column_id}]: {values[0]}"
    
    elif operation == "sum":
        result = sum(values)
        answer = str(int(result) if result == int(result) else result)
        trace = f"Sum of {len(values)} operands from {operands[0].table_id}: {' + '.join(str(v) for v in values)} = {result}"
    
    elif operation == "average":
        result = sum(values) / len(values)
        answer = str(int(result) if result == int(result) else result)
        trace = f"Average of {len(values)} operands: {result:.1f}"
    
    elif operation == "difference":
        if len(values) < 2:
            return "INSUFFICIENT_CONTEXT", "Need 2 operands for difference"
        result = values[1] - values[0]
        answer = str(int(result) if result == int(result) else result)
        trace = f"Difference: {values[1]} - {values[0]} = {result}"
    
    elif operation == "percentage":
        if len(values) < 2:
            return "INSUFFICIENT_CONTEXT", "Need 2 operands for percentage"
        result = (values[1] / values[0] * 100) if values[0] != 0 else 0
        answer = str(int(result) if result == int(result) else result)
        trace = f"Percentage: ({values[1]} / {values[0]}) * 100 = {result:.1f}%"
    
    else:
        return "INSUFFICIENT_CONTEXT", f"Unknown operation: {operation}"
    
    return answer, trace
