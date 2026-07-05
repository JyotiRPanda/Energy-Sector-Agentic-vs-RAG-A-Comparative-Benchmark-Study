"""
Structured table retrieval: two-stage pipeline that uses annotation CSVs.

Stage 1: Lexical search on corpus → identify candidate source_file(s)
Stage 2: SQL-style annotation-table lookup → get exact cell values

This replaces the broken "retrieve arbitrary cells and hope they are right"
approach with a schema-aware lookup that reads the original PDF table directly.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from gri_benchmark.annotation_index import AnnotationTableIndex, CellRef, _parse_numeric
from gri_benchmark.evidence import RetrievedEvidence, EvidenceRecord


# ---------------------------------------------------------------------------
# Question parsing helpers
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")

_OPERATION_KEYWORDS = {
    # Most specific first — checked in order
    "average":    ["average", "mean", "avg"],
    "difference": ["difference", "change", "delta", "increase by", "decrease by",
                   "reduction by", "grew by", "fell by"],
    "percentage": ["percent", "percentage", "ratio", "proportion"],
    "sum":        ["sum", "combined", "add", "adding", "sum of", "total of",
                   "total sum"],
    "extractive": [],          # fallback
}


def extract_years_from_question(question: str) -> List[str]:
    """Return sorted list of unique 4-digit years found in question."""
    return sorted(set(_YEAR_RE.findall(question)))


def infer_operation(question: str) -> str:
    """Detect the arithmetic operation from question text (word-boundary safe)."""
    q = question.lower()
    # Most-specific first; use word-boundary matching to avoid substring false-positives
    for op, keywords in _OPERATION_KEYWORDS.items():
        if op == "extractive":
            continue
        for kw in keywords:
            # Escape and wrap in word boundaries
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, q):
                return op
    return "extractive"


def extract_metric_keywords(question: str) -> str:
    """
    Pull metric-related keywords from the question by removing
    operation words, year mentions, and common filler words.
    Keeps longer meaningful tokens (GRI codes, metric names).
    """
    stopwords = {
        "what", "is", "the", "of", "in", "for", "and", "a", "an",
        "how", "much", "many",
        "sum", "average", "mean", "difference", "between",
        "percent", "percentage", "value", "total",
        "were", "was", "are", "by", "from", "to", "per", "with",
        "years", "year",
    }
    # Remove years
    q = _YEAR_RE.sub("", question.lower())
    tokens = re.findall(r"[a-z][a-z0-9]+", q)
    keywords = [t for t in tokens if t not in stopwords and len(t) > 2]
    return " ".join(keywords)


# ---------------------------------------------------------------------------
# Structured retriever
# ---------------------------------------------------------------------------

class StructuredTableRetriever:
    """
    Retrieves answers by looking up the actual annotation table CSVs.

    Given a question and a set of candidate source files (from initial
    lexical retrieval), this class performs a precise cell lookup:

      1. Parse question → years, operation, metric keywords
      2. For each candidate source_file, query AnnotationTableIndex
      3. Aggregate found cell values with the detected operation
      4. Return a RetrievedEvidence with the exact computed answer
         stored in primary_value, plus full provenance in score_breakdown.
    """

    def __init__(self, annotation_index: AnnotationTableIndex) -> None:
        self.index = annotation_index

    def retrieve(
        self,
        question: str,
        candidate_source_files: List[str],
        top_k: int = 1,
    ) -> List[RetrievedEvidence]:
        """
        Main entry point.

        Args:
            question:               The full question text.
            candidate_source_files: Source files from initial lexical retrieval.
            top_k:                  Max results to return.

        Returns:
            List of RetrievedEvidence, each with primary_value set to the
            computed answer and score_breakdown containing cell provenance.
        """
        years = extract_years_from_question(question)
        operation = infer_operation(question)
        metric_query = extract_metric_keywords(question)

        # Only attempt structured lookup when:
        # 1. Operation is sum or average (we reliably compute these)
        # 2. At least 2 explicit years in question (single-year lookups are
        #    handled better by the existing corpus retrieval)
        if operation not in ("sum", "average"):
            return []
        if len(years) < 2:
            return []

        results: List[Tuple[float, RetrievedEvidence]] = []

        for source_file in candidate_source_files:
            cells = self.index.query(
                source_file=source_file,
                metric_query=metric_query,
                years=years,
                operation=operation,
            )

            if not cells:
                continue

            # CONFIDENCE GATE: require all requested years found with numeric values
            years_found = {c.year for c in cells if c.year and c.numeric_value is not None}
            if len(years_found) < len(years):
                continue  # Partial year coverage → unreliable

            # All cells must be from the same row (same metric)
            row_ids = {(c.row_idx, c.page_nbr, c.table_nbr) for c in cells}
            if len(row_ids) > 1:
                continue  # Cells from different rows → incoherent

            # Aggregate cell values
            computed = self.index.aggregate(cells, operation)
            if computed is None:
                continue

            answer_str = self.index.format_result(computed)

            # Coverage is now always 1.0 (we required all years above)
            coverage = 1.0

            # Build a synthetic EvidenceRecord representing the answer
            primary_cell = cells[0]
            record = EvidenceRecord(
                record_id=f"struct-{primary_cell.pdf_name}-p{primary_cell.page_nbr}-t{primary_cell.table_nbr}-r{primary_cell.row_idx}-c{primary_cell.col_idx}",
                split="structured_lookup",
                source_file=source_file,
                table_id=primary_cell.table_nbr,
                row_id=str(primary_cell.row_idx),
                column_id=str(primary_cell.col_idx),
                primary_value=answer_str,
                content_text=(
                    f"[STRUCTURED LOOKUP] {primary_cell.row_label} | "
                    f"years={years} | op={operation} | result={answer_str}"
                ),
                years=tuple(y for y in years if y),
                units=(),
                intents=(operation,),
                domain="structured",
            )

            score = 0.6 + 0.4 * coverage   # base confidence

            ev = RetrievedEvidence(
                record=record,
                score=score,
                score_breakdown={
                    "mode": "structured_lookup",
                    "operation": operation,
                    "years_requested": years,
                    "years_found": [c.year for c in cells],
                    "metric_query": metric_query,
                    "cells": [
                        {
                            "row_label": c.row_label,
                            "col_label": c.col_label,
                            "year": c.year,
                            "raw_value": c.value,
                            "numeric_value": c.numeric_value,
                            "page_nbr": c.page_nbr,
                            "table_nbr": c.table_nbr,
                            "row_idx": c.row_idx,
                            "col_idx": c.col_idx,
                        }
                        for c in cells
                    ],
                    "computed_answer": answer_str,
                    "coverage": coverage,
                },
            )
            results.append((score, ev))

        results.sort(key=lambda x: -x[0])
        return [ev for _, ev in results[:top_k]]
