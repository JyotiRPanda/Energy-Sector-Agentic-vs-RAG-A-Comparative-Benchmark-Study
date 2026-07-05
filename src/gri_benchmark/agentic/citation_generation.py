"""Citation generation for agentic pipeline.

Generates citations from evidence records that were actually used
to produce answers, not just from the top retrieved hit.
"""

from typing import Optional

from gri_benchmark.evidence import EvidenceRecord, RetrievedEvidence
from gri_benchmark.types import Citation


def build_citation_from_record(
    record: EvidenceRecord,
    reason_used: str = "answer_source",
) -> Citation:
    """Build a citation from an evidence record.
    
    Args:
        record: EvidenceRecord to cite
        reason_used: Why this record was used (e.g., "calculation_input", "verified_selection", "lookup_result")
    
    Returns:
        Citation with all available fields populated
    """
    # Extract pdf_name and page_nbr from source_file if formatted as "pdf_name_page_X"
    pdf_name = None
    page_nbr = None
    if record.source_file:
        source_lower = str(record.source_file).lower()
        # Try to extract page number (e.g., "source_page_5" or "page_5")
        if "_page_" in source_lower:
            parts = source_lower.split("_page_")
            if len(parts) == 2:
                pdf_name = parts[0]
                try:
                    page_nbr = parts[1].replace(".pdf", "").replace(".txt", "").strip()
                except (ValueError, IndexError):
                    pdf_name = record.source_file
                    page_nbr = None
        else:
            pdf_name = record.source_file
    
    return Citation(
        source_file=str(record.source_file or ""),
        table_id=record.table_id,
        row_id=record.row_id,
        column_id=record.column_id,
        pdf_name=pdf_name,
        page_nbr=page_nbr,
        table_nbr=record.table_id,  # Using table_id as table_nbr
        primary_value=record.primary_value,
        evidence_id=record.record_id,
        reason_used=reason_used,
    )


def build_citations_from_calculation_trace(
    calculation_trace: dict,
    retrieved: list[RetrievedEvidence],
) -> list[Citation]:
    """Build citations from calculation trace.
    
    For calculations, cite all cells used as inputs.
    
    Args:
        calculation_trace: Calculation metadata with source_evidence list
        retrieved: All retrieved evidence for lookup
    
    Returns:
        List of citations for all input cells
    """
    citations = []
    
    if not calculation_trace or not calculation_trace.get("success"):
        return citations
    
    source_evidence = calculation_trace.get("source_evidence", [])
    
    # Build citations for each input cell
    for evidence_item in source_evidence:
        record_id = evidence_item.get("record_id")
        
        # Find the corresponding record
        matching_record = None
        for hit in retrieved:
            if hit.record.record_id == record_id:
                matching_record = hit.record
                break
        
        if matching_record:
            citation = build_citation_from_record(
                matching_record,
                reason_used=f"calculation_input ({evidence_item.get('value', 'N/A')})",
            )
            citations.append(citation)
    
    return citations


def build_citations_from_retrieved(
    retrieved: list[RetrievedEvidence],
    selected_record_id: Optional[str] = None,
    strategy: str = "extractive_lookup",
    calculation_trace: Optional[dict] = None,
) -> list[Citation]:
    """Build citations from retrieved evidence based on strategy.
    
    Args:
        retrieved: Retrieved evidence hits
        selected_record_id: Record ID of the selected/verified evidence
        strategy: Question strategy (extractive_lookup, relational_comparison, etc.)
        calculation_trace: Calculation metadata if available
    
    Returns:
        List of citations to include in prediction
    """
    citations = []
    
    if not retrieved:
        return citations
    
    # For calculations, cite all input sources
    if strategy in ["quantitative_calculation", "multistep_reasoning"] and calculation_trace:
        calculation_citations = build_citations_from_calculation_trace(calculation_trace, retrieved)
        if calculation_citations:
            return calculation_citations
    
    # For relational comparison, cite top candidates
    if strategy == "relational_comparison":
        # Cite top 2-3 candidates for comparison
        for hit in retrieved[:3]:
            citation = build_citation_from_record(
                hit.record,
                reason_used="comparison_candidate",
            )
            citations.append(citation)
        return citations
    
    # If a specific record was selected/verified, cite it
    if selected_record_id:
        for hit in retrieved:
            if hit.record.record_id == selected_record_id:
                citation = build_citation_from_record(
                    hit.record,
                    reason_used="selected_and_verified",
                )
                citations.append(citation)
                return citations
    
    # Default: cite the top retrieved hit
    if retrieved:
        citation = build_citation_from_record(
            retrieved[0].record,
            reason_used="top_retrieval",
        )
        citations.append(citation)
    
    return citations
