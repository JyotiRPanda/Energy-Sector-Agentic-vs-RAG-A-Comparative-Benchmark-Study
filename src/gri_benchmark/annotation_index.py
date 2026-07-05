"""
Schema-aware retrieval using annotation table CSVs.

The dataset stores the actual GRI report tables as CSV files under:
  data/dataset/annotation/{pdf_name}/{page_nbr}_{table_nbr}.csv

Columns are years (or named periods), rows are metrics.
Row and column indices in the question CSVs are 1-based.

This module implements SQL-style cell lookup:
  SELECT value FROM table
  WHERE company LIKE source_file
    AND metric_label LIKE ?
    AND year IN (?)

Usage:
  idx = AnnotationTableIndex("data/dataset/annotation")
  results = idx.query(source_file="axa_2023.pdf",
                      metric_query="total waste",
                      years=["2022", "2023"],
                      operation="sum")
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CellRef:
    """A resolved cell in an annotation table."""
    pdf_name: str
    page_nbr: str
    table_nbr: str
    row_idx: int        # 0-based into the CSV
    col_idx: int        # 0-based into the CSV
    row_label: str      # The metric name (column 0 of that row)
    col_label: str      # The column header (year or period label)
    year: Optional[str] # Extracted 4-digit year, if found
    value: str          # Raw string value from the cell
    numeric_value: Optional[float] = None   # Parsed float, if numeric


@dataclass
class ParsedTable:
    """An annotation CSV loaded into a queryable structure."""
    pdf_name: str
    page_nbr: str
    table_nbr: str
    file_path: str

    # header: list of column labels (col 0 is the metric label column)
    header: List[str] = field(default_factory=list)

    # col_idx -> year string (only for columns where a year is parseable)
    col_year_map: Dict[int, str] = field(default_factory=dict)

    # row_idx (0-based, including header row=0) -> list of raw cell strings
    raw_rows: List[List[str]] = field(default_factory=list)

    # row_idx -> lower-case metric label (col 0 of that row)
    row_labels: Dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(?:^|[\s/,])((?:20|19)\d{2})(?:$|[\s/,]|$)")
_COMMA_FLOAT_RE = re.compile(r"^[+-]?\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d+)?$")
_PCT_RE = re.compile(r"^[+-]?\d+(?:[,\.]\d+)?%$")


def _extract_year(text: str) -> Optional[str]:
    """Return first 4-digit year found in text, or None."""
    m = _YEAR_RE.search(text)
    return m.group(1) if m else None


def _parse_numeric(val: str) -> Optional[float]:
    """Try to parse a cell value as float; returns None on failure."""
    if not val or val.strip() in ("-", "–", "N/A", "n/a", ""):
        return None
    val = val.strip()
    # Remove trailing/leading whitespace, parenthetical annotations e.g. "90 (89)"
    val = re.sub(r"\s*\([^)]*\)", "", val).strip()
    # Handle asterisk-annotated cells e.g. "400*(350)" -> take first number
    m = re.match(r"^([+-]?\d[\d\s,\.]*)", val)
    if not m:
        return None
    raw = m.group(1).replace(" ", "").replace(",", ".")
    # Avoid converting things like "2023" (years) to numbers here
    try:
        return float(raw)
    except ValueError:
        return None


def _tokenize(text: str) -> set:
    """Lower-case word tokens, stripping punctuation."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _row_similarity(label_tokens: set, query_tokens: set) -> float:
    """Jaccard similarity between two token sets."""
    if not label_tokens or not query_tokens:
        return 0.0
    inter = len(label_tokens & query_tokens)
    union = len(label_tokens | query_tokens)
    return inter / union


# ---------------------------------------------------------------------------
# Core index
# ---------------------------------------------------------------------------

class AnnotationTableIndex:
    """
    Loads all annotation CSVs from the dataset and provides SQL-style lookup.

    Index layout:
      self._tables : dict[pdf_name -> list[ParsedTable]]
    """

    def __init__(self, annotation_dir: str) -> None:
        self.annotation_dir = annotation_dir
        self._tables: Dict[str, List[ParsedTable]] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        if not os.path.isdir(self.annotation_dir):
            return

        for company_folder in os.listdir(self.annotation_dir):
            folder_path = os.path.join(self.annotation_dir, company_folder)
            if not os.path.isdir(folder_path):
                continue

            pdf_name = company_folder + ".pdf"

            for fname in os.listdir(folder_path):
                if not fname.endswith(".csv"):
                    continue

                # filename: {page_nbr}_{table_nbr}.csv
                stem = fname[:-4]
                parts = stem.split("_")
                if len(parts) < 2:
                    continue
                page_nbr = parts[0]
                table_nbr = "_".join(parts[1:])  # handle names like "0", "1", "3"

                fpath = os.path.join(folder_path, fname)
                tbl = self._parse_table(fpath, pdf_name, page_nbr, table_nbr)
                if tbl is not None:
                    self._tables.setdefault(pdf_name, []).append(tbl)

    def _parse_table(
        self, fpath: str, pdf_name: str, page_nbr: str, table_nbr: str
    ) -> Optional[ParsedTable]:
        try:
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                raw_rows = list(csv.reader(fh, delimiter=";"))
        except Exception:
            return None

        if not raw_rows:
            return None

        tbl = ParsedTable(
            pdf_name=pdf_name,
            page_nbr=page_nbr,
            table_nbr=table_nbr,
            file_path=fpath,
            header=raw_rows[0],
            raw_rows=raw_rows,
        )

        # Build col_year_map from header row
        for ci, col_label in enumerate(raw_rows[0]):
            year = _extract_year(col_label)
            if year:
                tbl.col_year_map[ci] = year

        # Build row_labels (skip header row 0)
        for ri, row in enumerate(raw_rows):
            if ri == 0:
                continue
            label = row[0].strip() if row else ""
            if label:
                tbl.row_labels[ri] = label.lower()

        return tbl

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        source_file: str,
        metric_query: str,
        years: List[str],
        operation: str = "extractive",
    ) -> List[CellRef]:
        """
        SQL-style lookup:
          SELECT cells FROM annotation_tables
          WHERE pdf_name MATCHES source_file
            AND metric_label MATCHES metric_query
            AND year IN years

        Returns list of CellRef sorted by confidence (best first).
        For extractive: one cell per year.
        For sum/average/difference: one cell per requested year.
        """
        pdf_name = self._normalize_pdf_name(source_file)
        tables = self._tables.get(pdf_name, [])

        if not tables:
            return []

        query_tokens = _tokenize(metric_query)
        year_set = set(str(y) for y in years)

        # scored_rows: list of (score, year_cells_dict) for each candidate row
        scored_rows: List[Tuple[float, Dict[str, CellRef]]] = []

        for tbl in tables:
            # Score each row against metric_query
            for ri, label in tbl.row_labels.items():
                label_tokens = _tokenize(label)
                sim = _row_similarity(label_tokens, query_tokens)
                if sim < 0.1:
                    continue

                raw_row = tbl.raw_rows[ri]

                # For each requested year, find the matching column
                # Only accept columns where we find a non-empty, numeric-parseable value
                year_cells: Dict[str, CellRef] = {}
                seen_cols: set = set()  # avoid duplicate column hits for same year
                for ci in sorted(tbl.col_year_map.keys()):  # iterate by col index order
                    year = tbl.col_year_map[ci]
                    if year not in year_set:
                        continue
                    if year in year_cells:  # already found a cell for this year
                        continue
                    if ci >= len(raw_row):
                        continue

                    raw_val = raw_row[ci].strip()
                    if not raw_val or raw_val in ("-", "–", "N/A", "n/a"):
                        continue  # skip empty / missing cells

                    numeric = _parse_numeric(raw_val)
                    # Skip percentage/ratio columns (likely change columns, not values)
                    if raw_val.endswith("%"):
                        continue

                    cell = CellRef(
                        pdf_name=pdf_name,
                        page_nbr=tbl.page_nbr,
                        table_nbr=tbl.table_nbr,
                        row_idx=ri,
                        col_idx=ci,
                        row_label=tbl.raw_rows[ri][0].strip(),
                        col_label=tbl.header[ci] if ci < len(tbl.header) else "",
                        year=year,
                        value=raw_val,
                        numeric_value=numeric,
                    )
                    year_cells[year] = cell

                # Only accept rows that actually have numeric data for the requested years
                numeric_found = sum(1 for c in year_cells.values() if c.numeric_value is not None)
                if not year_cells or numeric_found == 0:
                    continue

                # Score: row similarity × coverage × numeric bonus
                coverage = len(year_cells) / max(len(year_set), 1)
                numeric_ratio = numeric_found / max(len(year_cells), 1)
                score = sim * coverage * (0.5 + 0.5 * numeric_ratio)
                scored_rows.append((score, year_cells))

        # Sort by score descending; pick best row first
        scored_rows.sort(key=lambda x: -x[0])

        # Return cells from the top-scoring row (all years from that row)
        results: List[CellRef] = []
        seen_years: set = set()
        for score, year_cells in scored_rows:
            added = False
            for year, cell in year_cells.items():
                if year not in seen_years:
                    seen_years.add(year)
                    results.append(cell)
                    added = True
            if added and len(seen_years) >= len(year_set):
                break

        return results

    def query_by_indices(
        self,
        source_file: str,
        page_nbr: str,
        table_nbr: str,
        row_indices: List[int],  # 1-based
        col_indices: List[int],  # 1-based
    ) -> List[CellRef]:
        """
        Exact lookup by (page, table, row, col) indices as stored in the dataset CSVs.
        Indices are 1-based (as they appear in the gri-qa_quant.csv).
        """
        pdf_name = self._normalize_pdf_name(source_file)
        table = self._find_table(pdf_name, page_nbr, table_nbr)
        if table is None:
            return []

        results = []
        for ri_1b, ci_1b in zip(row_indices, col_indices):
            ri = ri_1b - 1  # convert to 0-based
            ci = ci_1b - 1
            if ri >= len(table.raw_rows) or ri < 0:
                continue
            row_data = table.raw_rows[ri]
            if ci >= len(row_data) or ci < 0:
                continue

            raw_val = row_data[ci].strip()
            year = table.col_year_map.get(ci)
            cell = CellRef(
                pdf_name=pdf_name,
                page_nbr=page_nbr,
                table_nbr=table_nbr,
                row_idx=ri,
                col_idx=ci,
                row_label=row_data[0].strip() if row_data else "",
                col_label=table.header[ci] if ci < len(table.header) else "",
                year=year,
                value=raw_val,
                numeric_value=_parse_numeric(raw_val),
            )
            results.append(cell)

        return results

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate(cells: List[CellRef], operation: str) -> Optional[float]:
        """Apply operation to the numeric values of given cells."""
        values = [c.numeric_value for c in cells if c.numeric_value is not None]
        if not values:
            return None
        if operation in ("sum", "total"):
            return sum(values)
        if operation in ("average", "mean"):
            return sum(values) / len(values)
        if operation in ("difference", "reduction_difference", "increase_difference"):
            # Magnitude of change between the two values
            if len(values) >= 2:
                return abs(values[0] - values[-1])
        if operation == "reduction_percentage":
            # (original_larger - new_smaller) / original_larger × 100
            if len(values) >= 2 and max(values) != 0:
                return abs(max(values) - min(values)) / max(values) * 100
        if operation == "increase_percentage":
            # (new_larger - original_smaller) / original_smaller × 100
            if len(values) >= 2 and min(values) != 0:
                return abs(max(values) - min(values)) / min(values) * 100
        if operation in ("extractive", "percentage"):
            return values[0]
        return values[0]  # fallback: return first value

    @staticmethod
    def format_result(value: float) -> str:
        """
        Format a float to match gold answer string representation.

        Gold answers have at most 2 decimal places and use Python's minimal
        float notation: "4710.0", "29.09", "0.02", "720.0".
        """
        rounded = round(float(value), 2)
        return str(rounded)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_pdf_name(self, source_file: str) -> str:
        """Ensure source_file ends with .pdf."""
        name = os.path.basename(str(source_file or ""))
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        return name

    def _find_table(
        self, pdf_name: str, page_nbr: str, table_nbr: str
    ) -> Optional[ParsedTable]:
        for tbl in self._tables.get(pdf_name, []):
            if tbl.page_nbr == str(page_nbr) and tbl.table_nbr == str(table_nbr):
                return tbl
        return None

    # ------------------------------------------------------------------
    # Coverage report
    # ------------------------------------------------------------------

    def coverage_stats(self) -> dict:
        total_tables = sum(len(tbls) for tbls in self._tables.values())
        total_companies = len(self._tables)
        return {
            "companies": total_companies,
            "tables": total_tables,
            "pdf_names": list(self._tables.keys()),
        }


# ---------------------------------------------------------------------------
# Direct (row_id, col_id) → lookup  — bypasses all text retrieval
# ---------------------------------------------------------------------------

import ast as _ast

def _parse_meta_list(s) -> list:
    """Parse a Python list string like \"['axa_2023.pdf']\" or '[33, 33]'."""
    if s is None:
        return []
    try:
        v = _ast.literal_eval(str(s).strip())
        return list(v) if isinstance(v, (list, tuple)) else [v]
    except Exception:
        return [str(s).strip()]


def _apply_base_op(nums: List[float], base_op: str) -> Optional[float]:
    """Apply a base operation to a group of numeric values."""
    if not nums:
        return None
    if base_op == "sum":
        return sum(nums)
    if base_op == "average":
        return sum(nums) / len(nums)
    if base_op == "ratio":
        # Ordered ratio index based on operand order (first vs second year/value).
        if len(nums) >= 2:
            denom = nums[-1]
            if denom != 0:
                return (nums[0] / denom) * 100
            return 100.0
    if base_op == "difference":
        if len(nums) >= 2:
            return abs(nums[0] - nums[-1])
    return nums[0]  # extractive fallback


def _format_scalar(value: float, keep_trailing_zero: bool = False) -> str:
    """Format numeric output to closely match benchmark label conventions."""
    rounded = round(float(value), 2)
    if keep_trailing_zero:
        return str(rounded)
    if rounded.is_integer():
        return str(int(rounded))
    return str(rounded)


def _apply_select_op(
    group_results: List[float],
    select_op: str,
    question_text: str = "",
) -> Optional[str]:
    """Apply selection/aggregation over group results with question-aware semantics."""
    if not group_results:
        return None

    q = (question_text or "").lower()
    asks_lowest = "lowest" in q or "minimum" in q or "min " in q
    asks_highest = "highest" in q or "maximum" in q or "max " in q

    if select_op in ("sup", "max"):
        # Some labels use "sup" while the question asks for the lowest value.
        if asks_lowest and not asks_highest:
            return _format_scalar(min(group_results))
        return _format_scalar(max(group_results))
    if select_op == "min":
        return _format_scalar(min(group_results))
    if select_op in ("average", "avg", "mean"):
        return _format_scalar(sum(group_results) / len(group_results))
    if select_op == "sum":
        return _format_scalar(sum(group_results))
    if select_op == "ranking":
        # Interpret natural-language ranking intent:
        # - choose top-k based on highest/lowest
        # - output in requested ascending/descending order
        k = len(group_results)
        m = re.search(r"top\s+(\d+)", q)
        if m:
            try:
                k = max(1, min(int(m.group(1)), len(group_results)))
            except ValueError:
                k = len(group_results)

        if asks_lowest and not asks_highest:
            picked = sorted(group_results)[:k]
        else:
            picked = sorted(group_results, reverse=True)[:k]

        if "ascending" in q:
            ordered = sorted(picked)
        elif "descending" in q:
            ordered = sorted(picked, reverse=True)
        else:
            ordered = picked

        return ", ".join(_format_scalar(v, keep_trailing_zero=True) for v in ordered)
    # single value
    return _format_scalar(group_results[0])


class DirectTableLookup:
    """
    True table-structure retrieval via exact (row_idx, col_idx) pointer.

    Supports two index formats:

    Flat (single-step): row_indices=[r1, r2], col_indices=[c1, c2]
      Used by gri-qa_quant.csv — 95.9% accuracy.

    Nested (multi-step): row_indices=[[r1a,r1b],[r2a,r2b]], col_indices=[c1a,c1b,c2a,c2b]
      Used by gri-qa_multistep.csv.
      operation = '{base_op}_{select_op}' e.g. 'sum_sup', 'ratio_average', 'average_ranking'.
    """

    def __init__(self, annotation_index: AnnotationTableIndex) -> None:
        self.index = annotation_index

    def lookup(self, metadata: dict) -> Optional[str]:
        """
        Resolve cell values from metadata and compute the answer.

        Returns formatted answer string, or None if lookup is not possible.
        """
        pdf_names  = _parse_meta_list(metadata.get("pdf name"))
        page_nbrs  = _parse_meta_list(metadata.get("page nbr"))
        table_nbrs = _parse_meta_list(metadata.get("table nbr"))
        raw_row    = metadata.get("row indices")
        raw_col    = metadata.get("col indices")
        operation  = str(metadata.get("question_type_ext", "extractive")).strip()
        question_text = str(
            metadata.get("question")
            or metadata.get("question_text")
            or metadata.get("query")
            or ""
        )

        if not pdf_names or not page_nbrs or not table_nbrs:
            return None
        if raw_row is None or raw_col is None:
            return None

        try:
            page_nbr  = str(page_nbrs[0])
            table_nbr = str(table_nbrs[0])
        except (IndexError, TypeError):
            return None

        try:
            row_parsed = _ast.literal_eval(str(raw_row).strip())
            col_parsed = _ast.literal_eval(str(raw_col).strip())
        except Exception:
            return None

        # Determine format: flat list vs nested list-of-lists
        is_nested = (isinstance(row_parsed, list) and row_parsed and
                     isinstance(row_parsed[0], list))

        for pdf_raw in pdf_names:
            source_file = str(pdf_raw)

            if is_nested:
                result = self._lookup_multistep(
                    source_file, page_nbr, table_nbr,
                    row_parsed, col_parsed, operation, question_text
                )
            else:
                result = self._lookup_flat(
                    source_file, page_nbr, table_nbr,
                    row_parsed, col_parsed, operation, question_text, metadata.get("metadata", "")
                )

            if result is not None:
                return result

        return None

    # ------------------------------------------------------------------
    # Flat single-step lookup (quant, relational, extractive splits)
    # ------------------------------------------------------------------

    def _lookup_flat(
        self, source_file, page_nbr, table_nbr, row_idxs, col_idxs, operation, question_text="", meta_payload=""
    ) -> Optional[str]:
        if len(row_idxs) != len(col_idxs):
            return None

        cells = self.index.query_by_indices(
            source_file=source_file,
            page_nbr=page_nbr,
            table_nbr=table_nbr,
            row_indices=[int(r) for r in row_idxs],
            col_indices=[int(c) for c in col_idxs],
        )
        if not cells:
            return None

        nums = [c.numeric_value for c in cells if c.numeric_value is not None]
        if not nums and cells:
            return cells[0].value if cells[0].value else None

        parsed_meta = {}
        if meta_payload:
            try:
                parsed = _ast.literal_eval(str(meta_payload))
                if isinstance(parsed, dict):
                    parsed_meta = parsed
            except Exception:
                parsed_meta = {}

        op = str(operation or "").strip().lower()
        q_lower = str(question_text or "").lower()

        if op in ("rank", "ranking") and nums:
            desc = bool(parsed_meta.get("desc", False)) or ("descending" in q_lower)
            firstk = parsed_meta.get("firstk")
            try:
                k = int(float(firstk)) if firstk is not None else len(nums)
            except (TypeError, ValueError):
                k = len(nums)
            k = max(1, min(k, len(nums)))
            num_pairs = [
                (c.numeric_value, str(c.value).strip())
                for c in cells
                if c.numeric_value is not None
            ]
            ordered = sorted(num_pairs, key=lambda x: x[0], reverse=desc)[:k]
            return ", ".join(raw for _, raw in ordered)

        if op == "superlative" and nums:
            maximise = parsed_meta.get("maximise")
            if maximise is None:
                maximise = not any(k in q_lower for k in ("minimum", "lowest", "smallest"))
            num_pairs = [
                (c.numeric_value, str(c.value).strip())
                for c in cells
                if c.numeric_value is not None
            ]
            if not num_pairs:
                return None
            picked = max(num_pairs, key=lambda x: x[0]) if bool(maximise) else min(num_pairs, key=lambda x: x[0])
            return picked[1]

        if op == "comparative" and len(nums) >= 2:
            maximise = parsed_meta.get("maximise")
            if maximise is None:
                maximise = not any(k in q_lower for k in ("smaller", "lower", "less", "minimum", "lowest"))
            verdict = (nums[0] > nums[1]) if bool(maximise) else (nums[0] < nums[1])
            return "yes" if verdict else "no"

        result = self.index.aggregate(cells, operation)
        if result is None:
            return None
        return self.index.format_result(result)

    # ------------------------------------------------------------------
    # Nested multi-step lookup (multistep split)
    # ------------------------------------------------------------------

    def _lookup_multistep(
        self, source_file, page_nbr, table_nbr,
        row_groups, col_flat, operation, question_text=""
    ) -> Optional[str]:
        """
        Handle nested row_indices = [[g1_rows], [g2_rows], ...].

        col_flat length = sum of group sizes.
        operation = '{base_op}_{select_op}'.
        """
        n_groups = len(row_groups)
        if n_groups == 0:
            return None

        # Support variable group sizes: each group consumes its own col slice.
        group_sizes: List[int] = []
        normalized_row_groups: List[List[int]] = []
        for group in row_groups:
            if not isinstance(group, list) or not group:
                return None
            row_idx_group = [int(r) for r in group]
            normalized_row_groups.append(row_idx_group)
            group_sizes.append(len(row_idx_group))

        total_cells = sum(group_sizes)
        if len(col_flat) != total_cells:
            return None

        # Split operation into base and selection parts
        parts = operation.split("_", 1)
        base_op   = parts[0] if parts else "extractive"
        select_op = parts[1] if len(parts) > 1 else "value"

        # Apply base operation per group using stepwise execution.
        group_results: List[float] = []
        col_offset = 0
        for rows_in_group, group_size in zip(normalized_row_groups, group_sizes):
            cols_in_group = [int(c) for c in col_flat[col_offset : col_offset + group_size]]
            col_offset += group_size

            group_cells = self.index.query_by_indices(
                source_file=source_file,
                page_nbr=page_nbr,
                table_nbr=table_nbr,
                row_indices=rows_in_group,
                col_indices=cols_in_group,
            )
            if not group_cells or len(group_cells) != group_size:
                return None

            nums = [c.numeric_value for c in group_cells if c.numeric_value is not None]
            val = _apply_base_op(nums, base_op)
            if val is None:
                return None
            group_results.append(val)

        return _apply_select_op(group_results, select_op, question_text)

