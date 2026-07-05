from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from gri_benchmark.types import BenchmarkExample


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _first_non_empty(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip().strip("\"'")
        if text:
            return text
    return ""


def _extract_years(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", text))))


def _extract_units(text: str) -> tuple[str, ...]:
    """Extract and normalize energy/mass units from text.
    
    Normalizes variants:
    - GWh, MWh, GJ (energy)
    - tons, ton, t, T (mass)
    - m3, m^3, m³ (volume)
    - %, percent, percentage (ratio)
    """
    lowered = text.lower()
    found = set()
    padded = f" {lowered} "
    
    # Energy units
    if "gwh" in lowered:
        found.add("gwh")
    if "mwh" in lowered:
        found.add("mwh")
    if "gj" in lowered:
        found.add("gj")
    
    # Mass/weight units (normalize all variants to "tons")
    if any(m in lowered for m in ["tons", "ton"]):
        found.add("tons")
    if " t " in padded or "-t-" in lowered or "t " in lowered:
        found.add("tons")
    
    # Volume units (normalize to "m3")
    if any(m in lowered for m in ["m3", "m^3", "m³"]):
        found.add("m3")
    
    # Percentage/ratio units
    if "%" in lowered or "percent" in lowered:
        found.add("percent")
    
    return tuple(sorted(found))


def _extract_intents(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    intents = set()
    intent_map = {
        # Aggregation
        "average": "average",
        "mean": "average",
        "median": "average",
        "sum": "sum",
        "total": "sum",
        "combined": "sum",
        "aggregate": "sum",
        # Difference / change
        "difference": "difference",
        "increase": "difference",
        "decrease": "difference",
        "reduction": "difference",
        "change": "difference",
        "growth": "difference",
        "decline": "difference",
        "drop": "difference",
        "rise": "difference",
        "between": "difference",
        # Percentage / ratio
        "percentage": "percentage",
        "percent": "percentage",
        "ratio": "percentage",
        "proportion": "percentage",
        "fraction": "percentage",
        "rate": "percentage",
        # Superlatives
        "maximum": "superlative",
        "minimum": "superlative",
        "lowest": "superlative",
        "highest": "superlative",
        "largest": "superlative",
        "smallest": "superlative",
        "most": "superlative",
        "least": "superlative",
        "peak": "superlative",
        # Selection / comparison
        "which company": "company_selection",
        "which year": "company_selection",
        "rank": "ranking",
        "comparative": "comparative",
        "compared": "comparative",
        "versus": "comparative",
        "vs": "comparative",
        # Temporal ordering (for diff/pct questions)
        "year-on-year": "temporal",
        "yoy": "temporal",
        "year over year": "temporal",
        "prior year": "temporal",
        "previous year": "temporal",
        "prior period": "temporal",
        "compared to": "temporal",
    }
    for marker, label in intent_map.items():
        if marker in lowered:
            intents.add(label)

    # Pattern-based detection for percentage questions (e.g., "% change", "X% reduction")
    import re as _re
    if _re.search(r"\d+\s*%|\bpct\b|percentage\s+\w+|%\s+change", lowered):
        intents.add("percentage")

    # Detect explicit before/after ordering
    if _re.search(r"\b(before|after|from\s+\d{4}\s+to|in\s+\d{4}\s+vs)\b", lowered):
        intents.add("temporal")

    return tuple(sorted(intents))


def _extract_domain(text: str) -> str:
    """Classify chunk domain based on keywords.
    
    Domains:
    - emissions: GHG, CO2, scope, emissions, carbon
    - energy: energy, mwh, gwh, gj, consumption, renewable
    - water: water, withdrawn, discharge, m3, usage
    - waste: waste, hazardous, non-hazardous, disposal, recycling, landfill
    - biodiversity: biodiversity, species, habitat, land, forest
    - other: default
    """
    lowered = text.lower()
    
    # Emissions domain (highest priority - most specific)
    if any(w in lowered for w in ["ghg", "co2", "carbon", "emissions", "scope 1", "scope 2", "scope 3", "methane"]):
        return "emissions"
    
    # Energy domain
    if any(w in lowered for w in ["energy", "mwh", "gwh", "gj", "consumption", "renewable", "solar", "wind"]):
        return "energy"
    
    # Water domain
    if any(w in lowered for w in ["water", "withdrawn", "discharge", "m3", "seawater", "groundwater", "usage"]):
        return "water"
    
    # Waste domain
    if any(w in lowered for w in ["waste", "hazardous", "disposal", "recycling", "landfill", "kilotons"]):
        return "waste"
    
    # Biodiversity domain
    if any(w in lowered for w in ["biodiversity", "species", "habitat", "land", "forest", "iucn red list"]):
        return "biodiversity"
    
    return "other"


def _calculate_domain_bonus(
    query_text: str,
    chunk_domain: str,
    domain_penalty: float = -0.1,
    reward_only: bool = False,
) -> float:
    """Calculate domain relevance bonus (0.0 to 0.3).

    Modes:
      reward_only=False (default): +0.3 match, domain_penalty mismatch
      reward_only=True            : +0.3 match, 0.0 mismatch (no penalty)

    Args:
        query_text: The query string.
        chunk_domain: Domain classification of the retrieved chunk.
        domain_penalty: Penalty applied when reward_only=False. Default -0.1.
                        Use -0.05 for the weak-penalty ablation variant.
        reward_only: When True, mismatching domains are not penalised at all.
                     This preserves cross-domain context diversity while still
                     boosting on-domain chunks.
    """
    query_domain = _extract_domain(query_text)

    if query_domain == "other" or chunk_domain == "other":
        return 0.0  # No bonus/penalty for unknown domains

    if query_domain == chunk_domain:
        return 0.3  # Strong bonus for same domain

    return 0.0 if reward_only else domain_penalty


@dataclass(frozen=True)
class EvidenceRecord:
    record_id: str
    split: str
    source_file: str | None
    table_id: str | None
    row_id: str | None
    column_id: str | None
    primary_value: str | None
    content_text: str
    years: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    domain: str = "other"  # emissions, energy, water, waste, biodiversity, other
    is_table_data: bool = False  # True if from actual table cell (has row_id/column_id)
    metric_type: str = "unknown"  # waste, energy, emissions, water, other (from table metadata)
    metric_type: str = "unknown"  # waste, energy, emissions, water, other (from table metadata)


@dataclass(frozen=True)
class RetrievedEvidence:
    record: EvidenceRecord
    score: float
    score_breakdown: dict[str, float]


class SimpleEvidenceRetriever:
    def __init__(self, records: list[EvidenceRecord], use_domain_aware: bool = False):
        self.records = records
        self._record_tokens = [_tokenize(rec.content_text) for rec in records]
        self.use_domain_aware = use_domain_aware
        self.table_metadata = None  # Loaded on demand
        self.metric_lookup: dict[tuple[str, str], str] = {}  # (table_id, row_id) -> metric_type
        self.table_metadata = None  # Loaded on demand
        self.metric_lookup: dict[tuple[str, str], str] = {}  # (table_id, row_id) -> metric_type

    @classmethod
    def from_examples(cls, examples: list[BenchmarkExample]) -> "SimpleEvidenceRetriever":
        records: list[EvidenceRecord] = []
        for ex in examples:
            md = ex.metadata
            primary_value = _first_non_empty(md.get("value"), md.get("answer_value"), ex.gold_answer) or None
            source_file = _first_non_empty(md.get("source_file"), md.get("source")) or None
            table_id = _first_non_empty(md.get("table_id")) or None
            row_id = _first_non_empty(md.get("row_id")) or None
            column_id = _first_non_empty(md.get("column_id")) or None

            content_text = " ".join(
                part
                for part in [
                    ex.question,
                    _first_non_empty(md.get("gri")),
                    _first_non_empty(md.get("question_type_ext")),
                    _first_non_empty(md.get("pdf name")),
                    _first_non_empty(source_file),
                    _first_non_empty(table_id),
                    _first_non_empty(primary_value),
                ]
                if part
            )

            years = _extract_years(content_text)
            units = _extract_units(content_text)
            intents = _extract_intents(content_text)
            domain = _extract_domain(content_text)
            is_table_data = bool(row_id and column_id)  # Real table cell if has both row and column

            records.append(
                EvidenceRecord(
                    record_id=ex.question_id,
                    split=ex.split,
                    source_file=source_file,
                    table_id=table_id,
                    row_id=row_id,
                    column_id=column_id,
                    primary_value=primary_value,
                    content_text=content_text,
                    years=years,
                    units=units,
                    intents=intents,
                    domain=domain,
                    is_table_data=is_table_data,
                )
            )
        return cls(records)

    def load_table_metadata(self, annotation_root: Path, dataset_dir: Path | None = None) -> None:
        """Load table metadata to enrich evidence with metric types from source tables.
        
        This builds a mapping from (source_file, row_id) to metric_type by:
        1. Scanning dataset CSV files to find which PDF/page/table contains each row
        2. Loading table metadata from annotation directory
        3. Storing metric type for each (table_id, row_id) pair
        
        Args:
            annotation_root: Path to annotation directory containing table CSVs
            dataset_dir: Path to dataset directory with CSV files (for building PDF->table mappings)
        """
        try:
            from gri_benchmark.table_metadata import TableMetadataExtractor
        except ImportError:
            return  # Silently skip if table_metadata not available
        
        self.table_metadata = TableMetadataExtractor(Path(annotation_root))
        self._source_to_table_info: dict[tuple[str, str], tuple[str, int, int]] = {}  # (source_file, row_id) -> (pdf_name, page, table)
        
        # Step 1: Build mapping from source file + row ID to (pdf_name, page, table_num)
        if dataset_dir and dataset_dir.exists():
            self._build_source_row_to_table_mapping(Path(dataset_dir))
        
        # Step 2: Use mapping to populate metric_lookup
        for record in self.records:
            if record.is_table_data and record.source_file and record.row_id:
                key = (record.source_file, record.row_id)
                if key in self._source_to_table_info:
                    pdf_name, page_nbr, table_nbr = self._source_to_table_info[key]
                    try:
                        metadata = self.table_metadata.load_table(pdf_name, page_nbr, table_nbr)
                        if metadata:
                            metric = next((m for m in metadata.metrics if m.row_idx == int(record.row_id)), None)
                            if metric:
                                rec_key = (record.table_id, record.row_id)
                                self.metric_lookup[rec_key] = metric.metric_type
                    except (ValueError, AttributeError):
                        pass
    
    def _build_source_row_to_table_mapping(self, dataset_dir: Path) -> None:
        """Scan dataset CSVs to build mapping of (source_file, row_id) -> (pdf_name, page, table_num)."""
        import csv
        import ast
        
        for csv_path in dataset_dir.glob("**/*.csv"):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            pdf_list = ast.literal_eval(row.get('pdf name', '[]'))
                            page_list = ast.literal_eval(row.get('page nbr', '[]'))
                            table_list = ast.literal_eval(row.get('table nbr', '[]'))
                            row_indices = ast.literal_eval(row.get('row indices', '[]'))
                            
                            if pdf_list and page_list and table_list and row_indices:
                                pdf_file = pdf_list[0]  # e.g., "axa_2023.pdf"
                                pdf_name = pdf_file.replace('.pdf', '')
                                page_nbr = page_list[0]
                                table_nbr = table_list[0]
                                
                                # Map each row_id to this table
                                for row_idx in set(row_indices):
                                    key = (pdf_file, str(row_idx))
                                    self._source_to_table_info[key] = (pdf_name, page_nbr, table_nbr)
                        except Exception:
                            pass
            except Exception:
                pass

    @classmethod
    def from_jsonl(cls, corpus_path: str | Path) -> "SimpleEvidenceRetriever":
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        records: list[EvidenceRecord] = []
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            
            content_text = _first_non_empty(row.get("content_text"))
            row_id = _first_non_empty(row.get("row_id")) or None
            column_id = _first_non_empty(row.get("column_id")) or None
            
            years = tuple(row.get("years", []))
            units = tuple(row.get("units", []))
            intents = tuple(row.get("intents", []))
            domain = row.get("domain") or _extract_domain(content_text)
            is_table_data = bool(row_id and column_id)
            
            records.append(
                EvidenceRecord(
                    record_id=str(row.get("record_id", f"record-{idx}")),
                    split=str(row.get("split", "eval")),
                    source_file=_first_non_empty(row.get("source_file")) or None,
                    table_id=_first_non_empty(row.get("table_id")) or None,
                    row_id=row_id,
                    column_id=column_id,
                    primary_value=_first_non_empty(row.get("primary_value")) or None,
                    content_text=content_text,
                    years=years,
                    units=units,
                    intents=intents,
                    domain=domain,
                    is_table_data=is_table_data,
                )
            )

        retriever = cls(records)
        
        # Load table metadata if annotation directory exists
        try:
            import os
            corpus_path_obj = Path(corpus_path).resolve()
            
            # Try multiple possible locations for annotation directory
            possible_annotation_roots = [
                corpus_path_obj.parent.parent / 'annotation',           # data/annotation
                corpus_path_obj.parent.parent / 'dataset' / 'annotation', # data/dataset/annotation
            ]
            
            annotation_root = None
            for path in possible_annotation_roots:
                if path.exists():
                    annotation_root = path
                    break
            
            if annotation_root:
                dataset_dir = corpus_path_obj.parent.parent / 'dataset'
                if not dataset_dir.exists():
                    # Check parent levels
                    dataset_dir = corpus_path_obj.parent.parent.parent / 'dataset'
                
                if dataset_dir.exists():
                    retriever.load_table_metadata(annotation_root, dataset_dir)
                else:
                    retriever.load_table_metadata(annotation_root)
        except Exception:
            pass  # Silently continue if metadata loading fails
        
        return retriever

    def search(
        self,
        query: str,
        *,
        split: str | None = None,
        source_file: str | None = None,
        top_k: int = 3,
        use_constraints: bool = True,
    ) -> list[RetrievedEvidence]:
        """Multi-stage constrained retrieval with explicit fallback.
        
        If use_domain_aware is True, delegates to search_with_domain_awareness.
        Otherwise uses lexical-only scoring.
        """
        if self.use_domain_aware:
            return self.search_with_domain_awareness(
                query,
                split=split,
                source_file=source_file,
                top_k=top_k,
                use_constraints=use_constraints,
            )
        
        # Original lexical-only search
        """Multi-stage constrained retrieval with explicit fallback.
        
        Stage A: Lexical scoring on all candidates
        Stage B: Filter to year-matching candidates (if query has years)
        Stage C: Filter to unit-matching candidates (if query has units)
        Stage D: Rerank survivors by lexical + intent match
        
        If any filtering stage empties the set, log fallback and relax one constraint.
        """
        query_tokens = _tokenize(query)
        query_years = set(_extract_years(query))
        query_units = set(_extract_units(query))
        query_intents = set(_extract_intents(query))
        
        scored: list[RetrievedEvidence] = []
        
        # ===== Stage A: Lexical scoring on all candidates =====
        candidates_stage_a = []
        for rec, rec_tokens in zip(self.records, self._record_tokens):
            if split and rec.split != split:
                continue

            overlap = len(query_tokens & rec_tokens)
            denom = max(len(query_tokens), 1)
            lexical_score = overlap / denom

            # Skip if no lexical overlap
            if overlap == 0 and lexical_score < 0.15:
                continue

            intent_match = 0.0
            if query_intents and rec.intents and (query_intents & set(rec.intents)):
                intent_match = 1.0

            candidates_stage_a.append({
                "record": rec,
                "rec_tokens": rec_tokens,
                "lexical_score": lexical_score,
                "intent_match": intent_match,
                "from_fallback": False,
                "fallback_reason": None,
            })
        
        if not use_constraints:
            # Raw retrieval mode: lexical + intent only, no year/unit gating.
            for c in candidates_stage_a:
                rec = c["record"]
                lexical_score = c["lexical_score"]
                intent_match = c["intent_match"]
                score = lexical_score + (0.15 * intent_match)
                if source_file and rec.source_file and source_file == rec.source_file:
                    score += 0.15

                scored.append(
                    RetrievedEvidence(
                        record=rec,
                        score=score,
                        score_breakdown={
                            "mode": "raw_retrieval",
                            "lexical": round(lexical_score, 6),
                            "intent_match": round(intent_match, 6),
                            "year_constraint_satisfied": True,
                            "unit_constraint_satisfied": True,
                            "from_fallback": False,
                            "fallback_reason": None,
                            "expected_years": list(query_years),
                            "expected_units": list(query_units),
                            "top_hit_years": list(rec.years) if rec.years else [],
                            "top_hit_units": list(rec.units) if rec.units else [],
                        },
                    )
                )

            scored.sort(key=lambda x: x.score, reverse=True)
            return scored[: max(top_k, 1)]

        # ===== Stage B: Year constraint filtering =====
        candidates_stage_b = candidates_stage_a
        year_constraint_satisfied = True
        
        if query_years:
            year_filtered = [
                c for c in candidates_stage_a
                if c["record"].years and (query_years & set(c["record"].years))
            ]
            
            if year_filtered:
                candidates_stage_b = year_filtered
            else:
                # Fallback: keep all, but mark as fallback
                year_constraint_satisfied = False
                candidates_stage_b = [
                    {**c, "from_fallback": True, "fallback_reason": "no_year_match"}
                    for c in candidates_stage_a
                ]
        
        # ===== Stage C: Unit constraint filtering =====
        candidates_stage_c = candidates_stage_b
        unit_constraint_satisfied = True
        
        if query_units:
            unit_filtered = [
                c for c in candidates_stage_b
                if c["record"].units and (query_units & set(c["record"].units))
            ]
            
            if unit_filtered:
                candidates_stage_c = unit_filtered
            else:
                # Fallback: relax unit constraint, keep all from stage B
                unit_constraint_satisfied = False
                candidates_stage_c = [
                    {**c, "from_fallback": True, "fallback_reason": c.get("fallback_reason") or "no_unit_match"}
                    for c in candidates_stage_b
                ]
        
        # ===== Stage D: Rerank by lexical + intent =====
        for c in candidates_stage_c:
            rec = c["record"]
            lexical_score = c["lexical_score"]
            intent_match = c["intent_match"]
            
            score = lexical_score + (0.15 * intent_match)
            
            if source_file and rec.source_file and source_file == rec.source_file:
                score += 0.15
            
            # Apply small penalty if from fallback, but don't exclude
            if c["from_fallback"]:
                score -= 0.05
            
            if score <= -0.5:
                continue
            
            scored.append(
                RetrievedEvidence(
                    record=rec,
                    score=score,
                    score_breakdown={
                        "lexical": round(lexical_score, 6),
                        "intent_match": round(intent_match, 6),
                        "year_constraint_satisfied": year_constraint_satisfied,
                        "unit_constraint_satisfied": unit_constraint_satisfied,
                        "from_fallback": c["from_fallback"],
                        "fallback_reason": c["fallback_reason"],
                        "expected_years": list(query_years),
                        "expected_units": list(query_units),
                        "top_hit_years": list(rec.years) if rec.years else [],
                        "top_hit_units": list(rec.units) if rec.units else [],
                    },
                )
            )
        
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: max(top_k, 1)]

    def search_with_domain_awareness(
        self,
        query: str,
        *,
        split: str | None = None,
        source_file: str | None = None,
        top_k: int = 3,
        use_constraints: bool = True,
        prioritize_table_data: bool = True,
        domain_penalty: float = -0.1,
        reward_only: bool = False,
    ) -> list[RetrievedEvidence]:
        """Enhanced retrieval with domain filtering and table prioritization.
        
        NEW STAGES:
        - Stage A: Lexical scoring (as before)
        - Stage B: Year constraint (as before)
        - Stage C: Unit constraint (as before)
        - Stage D: Domain filtering (NEW)
        - Stage E: Table data prioritization (NEW)
        - Stage F: Final rerank with domain/table bonuses (NEW)
        
        Domain filtering: If top results from different domain than query, 
        deprioritize them but don't exclude (allows fallback).
        
        Table prioritization: Table data (has row_id + column_id) scores higher
        than metadata-only chunks.
        """
        query_tokens = _tokenize(query)
        query_years = set(_extract_years(query))
        query_units = set(_extract_units(query))
        query_intents = set(_extract_intents(query))
        
        scored: list[RetrievedEvidence] = []
        
        # ===== Stage A: Lexical scoring on all candidates =====
        candidates_stage_a = []
        for rec, rec_tokens in zip(self.records, self._record_tokens):
            if split and rec.split != split:
                continue

            overlap = len(query_tokens & rec_tokens)
            denom = max(len(query_tokens), 1)
            lexical_score = overlap / denom

            # Skip if no lexical overlap
            if overlap == 0 and lexical_score < 0.15:
                continue

            intent_match = 0.0
            if query_intents and rec.intents and (query_intents & set(rec.intents)):
                intent_match = 1.0

            candidates_stage_a.append({
                "record": rec,
                "rec_tokens": rec_tokens,
                "lexical_score": lexical_score,
                "intent_match": intent_match,
                "from_fallback": False,
                "fallback_reason": None,
            })
        
        if not use_constraints:
            # Raw retrieval: don't apply domain filtering
            for c in candidates_stage_a:
                rec = c["record"]
                lexical_score = c["lexical_score"]
                intent_match = c["intent_match"]
                score = lexical_score + (0.15 * intent_match)
                if source_file and rec.source_file and source_file == rec.source_file:
                    score += 0.15

                scored.append(
                    RetrievedEvidence(
                        record=rec,
                        score=score,
                        score_breakdown={
                            "mode": "raw_retrieval_domain_aware",
                            "lexical": round(lexical_score, 6),
                            "intent_match": round(intent_match, 6),
                            "year_constraint_satisfied": True,
                            "unit_constraint_satisfied": True,
                            "domain_match": rec.domain,
                            "is_table_data": rec.is_table_data,
                            "from_fallback": False,
                            "fallback_reason": None,
                        },
                    )
                )

            scored.sort(key=lambda x: x.score, reverse=True)
            return scored[: max(top_k, 1)]

        # ===== Stage B: Year constraint filtering =====
        candidates_stage_b = candidates_stage_a
        year_constraint_satisfied = True
        
        if query_years:
            year_filtered = [
                c for c in candidates_stage_a
                if c["record"].years and (query_years & set(c["record"].years))
            ]
            
            if year_filtered:
                candidates_stage_b = year_filtered
            else:
                year_constraint_satisfied = False
                candidates_stage_b = [
                    {**c, "from_fallback": True, "fallback_reason": "no_year_match"}
                    for c in candidates_stage_a
                ]
        
        # ===== Stage C: Unit constraint filtering =====
        candidates_stage_c = candidates_stage_b
        unit_constraint_satisfied = True
        
        if query_units:
            unit_filtered = [
                c for c in candidates_stage_b
                if c["record"].units and (query_units & set(c["record"].units))
            ]
            
            if unit_filtered:
                candidates_stage_c = unit_filtered
            else:
                unit_constraint_satisfied = False
                candidates_stage_c = [
                    {**c, "from_fallback": True, "fallback_reason": c.get("fallback_reason") or "no_unit_match"}
                    for c in candidates_stage_b
                ]
        
        # ===== Stage D: Domain filtering (NEW) =====
        # Extract domain from first few candidates to see if we have domain hits
        has_domain_filtered = False
        domain_bonus_map = {}
        for c in candidates_stage_c:
            rec = c["record"]
            domain_bonus = _calculate_domain_bonus(
                query, rec.domain,
                domain_penalty=domain_penalty,
                reward_only=reward_only,
            )
            domain_bonus_map[id(rec)] = domain_bonus
            if domain_bonus > 0:
                has_domain_filtered = True
        
        # ===== Stage E: Table data prioritization (NEW) =====
        # Sort by: has_domain_match -> is_table_data -> lexical_score
        if prioritize_table_data:
            candidates_stage_c.sort(
                key=lambda c: (
                    domain_bonus_map[id(c["record"])] > 0,  # Domain match first
                    c["record"].is_table_data,  # Then table data
                    c["lexical_score"],  # Then lexical
                ),
                reverse=True
            )
        
        # ===== Stage F: Final rerank with bonuses =====
        for c in candidates_stage_c:
            rec = c["record"]
            lexical_score = c["lexical_score"]
            intent_match = c["intent_match"]
            
            score = lexical_score + (0.15 * intent_match)
            
            if source_file and rec.source_file and source_file == rec.source_file:
                score += 0.15
            
            # Add domain bonus (0.3 for match, -0.1 for mismatch, 0.0 for unknown)
            domain_bonus = domain_bonus_map[id(rec)]
            score += domain_bonus
            
            # Add table data bonus
            if rec.is_table_data:
                score += 0.1  # Prioritize table data
            
            # Apply fallback penalty
            if c["from_fallback"]:
                score -= 0.05
            
            if score <= -0.5:
                continue
            
            scored.append(
                RetrievedEvidence(
                    record=rec,
                    score=score,
                    score_breakdown={
                        "mode": "domain_aware",
                        "lexical": round(lexical_score, 6),
                        "intent_match": round(intent_match, 6),
                        "domain_bonus": round(domain_bonus, 6),
                        "table_bonus": round(0.1 if rec.is_table_data else 0.0, 6),
                        "year_constraint_satisfied": year_constraint_satisfied,
                        "unit_constraint_satisfied": unit_constraint_satisfied,
                        "query_domain": _extract_domain(query),
                        "chunk_domain": rec.domain,
                        "is_table_data": rec.is_table_data,
                        "from_fallback": c["from_fallback"],
                        "fallback_reason": c["fallback_reason"],
                        "expected_years": list(query_years),
                        "expected_units": list(query_units),
                        "top_hit_years": list(rec.years) if rec.years else [],
                        "top_hit_units": list(rec.units) if rec.units else [],
                    },
                )
            )
        
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: max(top_k, 1)]

    def search_with_row_joining(
        self,
        query: str,
        *,
        split: str | None = None,
        source_file: str | None = None,
        top_k: int = 3,
        use_constraints: bool = True,
        use_domain_aware: bool | None = None,
    ) -> list[RetrievedEvidence]:
        """
        JOIN-BASED RETRIEVAL: Restores relational context for table operands.
        
        Algorithm:
        1. Retrieve top-k chunks (lexical + constraints)
        2. For each chunk with table_id + row_id:
           a. Find ALL chunks with same (table_id, row_id)
           b. Group by column_id to reconstruct row context
           c. Augment primary_value with cell map
        3. Return results with complete row context
        
        This solves the core problem: Retrieved chunks from same row are disconnected.
        After joining, we have {2021: val1, 2022: val2, 2023: val3} instead of scattered cells.
        
        Example:
            Initial retrieve: waste 2022 (4710) ← isolated cell
            After joining:  {2021: 3908, 2022: 4710, 2023: 9420} ← complete row
        
        Args:
            query: Search query
            split: Optional dataset split filter
            source_file: Optional source file filter
            top_k: Number of top results to return
            use_constraints: Whether to apply year/unit constraints
            use_domain_aware: Use domain-aware retrieval if True (None defaults to self.use_domain_aware)
        
        Returns:
            List of RetrievedEvidence with row context enriched for table data
        """
        # Step 1: Use existing retrieval (respects constraints and domain)
        if use_domain_aware is None:
            use_domain_aware = self.use_domain_aware
        
        if use_domain_aware:
            initial_results = self.search_with_domain_awareness(
                query,
                split=split,
                source_file=source_file,
                top_k=top_k * 2,  # Get more to account for joining
                use_constraints=use_constraints,
            )
        else:
            initial_results = self.search(
                query,
                split=split,
                source_file=source_file,
                top_k=top_k * 2,  # Get more to account for joining
                use_constraints=use_constraints,
            )
        
        # Step 2: Build lookup for fast row retrieval
        # Index: (table_id, row_id) -> [records with that key]
        row_index: dict[tuple[str, str], list[EvidenceRecord]] = {}
        for rec in self.records:
            # Note: is_table_data defaults to False, so we check table identity directly
            if rec.table_id and rec.row_id:
                key = (rec.table_id, rec.row_id)
                if key not in row_index:
                    row_index[key] = []
                row_index[key].append(rec)
        
        # Step 3: Enrich results with row context
        enriched_results: list[RetrievedEvidence] = []
        seen_row_keys: set[tuple[str, str]] = set()  # Dedup by row key
        
        for retrieved in initial_results:
            rec = retrieved.record
            
            # If has table identity (table_id + row_id), join to full row
            # Note: is_table_data defaults to False, so we check table_id/row_id directly
            if rec.table_id and rec.row_id:
                row_key = (rec.table_id, rec.row_id)
                
                # Add metric type to score breakdown if available
                metric_type = self.metric_lookup.get(row_key, "unknown")
                retrieved.score_breakdown['metric_type'] = metric_type
                
                # Dedup: only include first occurrence of each row
                if row_key in seen_row_keys:
                    continue
                seen_row_keys.add(row_key)
                
                # Fetch all cells in this row
                row_cells = row_index.get(row_key, [])
                
                if row_cells:
                    # Build cell map: column_id -> values_by_year
                    # Parse comma-separated values if multiple years in cell
                    cell_map = {}
                    for cell in row_cells:
                        if not cell.column_id or not cell.primary_value:
                            continue
                        
                        # Parse comma-separated values
                        values_str = cell.primary_value.strip()
                        values_list = [v.strip() for v in values_str.split(',')]
                        years_list = list(cell.years) if cell.years else ['unknown']
                        
                        # Zip years with values
                        year_value_pairs = {}
                        for year, value_str in zip(years_list, values_list):
                            try:
                                year_value_pairs[year] = float(value_str)
                            except ValueError:
                                pass  # Skip non-numeric values
                        
                        cell_map[cell.column_id] = {
                            "values_by_year": year_value_pairs,  # {year: float_value}
                            "raw_value": cell.primary_value,
                            "years": cell.years,
                            "units": cell.units,
                            "record_id": cell.record_id,
                        }
                    
                    # Merge metadata from all cells in row
                    all_years = set()
                    all_units = set()
                    all_intents = set()
                    for cell in row_cells:
                        all_years.update(cell.years)
                        all_units.update(cell.units)
                        all_intents.update(cell.intents)
                    
                    # Create enriched record with row context
                    enriched_content = f"[ROW CONTEXT] {rec.row_id} in {rec.table_id}:\n"
                    for col_id, cell_data in sorted(cell_map.items()):
                        year_values = cell_data['values_by_year']
                        enriched_content += f"  {col_id}: {year_values}\n"
                    enriched_content += f"\nOriginal chunk: {rec.content_text}"
                    
                    enriched_record = EvidenceRecord(
                        record_id=rec.record_id,
                        split=rec.split,
                        source_file=rec.source_file,
                        table_id=rec.table_id,
                        row_id=rec.row_id,
                        column_id=rec.column_id,
                        primary_value=rec.primary_value,
                        content_text=enriched_content,
                        years=tuple(sorted(all_years)),
                        units=tuple(sorted(all_units)),
                        intents=tuple(sorted(all_intents)),
                        domain=rec.domain,
                        is_table_data=rec.is_table_data,
                    )
                    
                    # Store cell map in score breakdown for downstream use
                    enriched_score_breakdown = dict(retrieved.score_breakdown)
                    enriched_score_breakdown["row_join_cell_map"] = cell_map
                    enriched_score_breakdown["row_join_all_years"] = list(sorted(all_years))
                    enriched_score_breakdown["row_join_all_units"] = list(sorted(all_units))
                    enriched_score_breakdown["row_join_all_intents"] = list(sorted(all_intents))
                    enriched_score_breakdown["row_join_count"] = len(row_cells)
                    
                    enriched_results.append(
                        RetrievedEvidence(
                            record=enriched_record,
                            score=retrieved.score + 0.1,  # Bonus for row-joined results
                            score_breakdown=enriched_score_breakdown,
                        )
                    )
                else:
                    # No other cells in row, return as-is
                    enriched_results.append(retrieved)
            else:
                # Non-table data, return as-is
                enriched_results.append(retrieved)
        
        # Step 4: Return top-k, sorted by score
        enriched_results.sort(key=lambda x: x.score, reverse=True)
        return enriched_results[:top_k]
