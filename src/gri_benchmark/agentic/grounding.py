"""Grounding layer for agentic pipeline.

Enforces precision-first operand selection:
1. Extract all numeric candidates from retrieved chunks
2. Score candidates by relevance to question intent
3. Select operands deterministically (before tool execution)
4. Tools only process committed operands (no exploration)

This constrains agentic behavior to match GRI-QA expectations:
- Specific table cells as answers
- Exact operand selection
- Simple deterministic computation
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from gri_benchmark.evidence import RetrievedEvidence


@dataclass
class CandidateNumber:
    """A numeric candidate extracted from retrieved evidence."""
    value: float
    source_chunk: RetrievedEvidence
    domain: str  # emissions, energy, water, waste, biodiversity, other
    unit: str    # MWh, tons, m3, %, USD, index, etc.
    temporal: str  # 2022, 2023, YoY, period, etc.
    confidence: float  # 0.0-1.0, based on relevance to question
    source_text: str  # Text snippet where number was found
    chunk_index: int  # Position in retrieval ranking
    
    def __repr__(self) -> str:
        return f"CandidateNumber({self.value:.1f} [{self.unit}], {self.domain}, {self.temporal}, conf={self.confidence:.2f})"


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text."""
    if not text:
        return []
    
    # Find all numbers: integers, decimals, percentages, scientific notation
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[-+]?\d+%'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            # Handle percentages
            if match.endswith('%'):
                num = float(match[:-1])
            else:
                num = float(match)
            numbers.append(num)
        except ValueError:
            continue
    
    return numbers


def _extract_unit(text: str) -> str:
    """Extract likely unit from text."""
    text_lower = text.lower()
    
    # Common units in energy sector
    unit_patterns = {
        'MWh': r'\bMWh\b|\bmwh\b',
        'GWh': r'\bGWh\b|\bgwh\b',
        'kWh': r'\bkWh\b|\bkwh\b',
        'MW': r'\bMW\b|\bmw\b',
        'GW': r'\bGW\b|\bgw\b',
        'tons': r'\bton[s]?\b|\bton[s]?\s+CO2|tonnes',
        'm3': r'\bm[³3]\b|\bcubic\s+meter',
        'gallons': r'\bgallon[s]?\b',
        'liters': r'\bliter[s]?\b|\blitre[s]?\b',
        '%': r'%|percent',
        'USD': r'\$|USD|dollars',
        'index': r'\bindex\b',
    }
    
    for unit, pattern in unit_patterns.items():
        if re.search(pattern, text_lower):
            return unit
    
    return 'unknown'


def _extract_temporal(text: str) -> str:
    """Extract temporal reference from text."""
    text_lower = text.lower()
    
    # Years
    year_match = re.search(r'\b(20\d{2})\b', text)
    if year_match:
        return year_match.group(1)
    
    # Periods
    period_patterns = {
        'YoY': r'\byear[- ]?on[- ]?year\b|YoY',
        'QoQ': r'\bquarter[- ]?on[- ]?quarter\b|QoQ',
        'total': r'\btotal\b|cumulative',
        'average': r'\baverage\b|mean',
        'change': r'\bchange\b|increase|decrease|growth',
    }
    
    for period, pattern in period_patterns.items():
        if re.search(pattern, text_lower):
            return period
    
    return 'unspecified'


def _score_candidate_relevance(
    question: str,
    chunk: RetrievedEvidence,
    number: float,
    chunk_index: int,
) -> float:
    """Score how relevant a candidate number is to the question.
    
    Factors:
    - Chunk ranking (top chunks score higher)
    - Domain match (domain from question matches chunk domain)
    - Unit match (question mentions unit)
    - Magnitude (number is reasonable for context)
    - Temporal alignment (year matches question)
    """
    score = 0.5  # Base score
    
    # Factor 1: Chunk position (closer to top = higher score)
    position_score = 1.0 - (chunk_index / 10.0)  # Decay over 10 chunks
    score += position_score * 0.2
    
    # Factor 2: Domain relevance
    question_domain = _extract_domain_from_question(question)
    chunk_domain = chunk.record.domain if hasattr(chunk.record, 'domain') else 'other'
    if question_domain and question_domain != 'other':
        if question_domain == chunk_domain:
            score += 0.2  # Strong domain match
        elif chunk_domain == 'other':
            score += 0.05  # Weak match
    
    # Factor 3: Unit mentioned in question
    question_units = _extract_units_from_question(question)
    chunk_unit = _extract_unit(chunk.record.content_text)
    if any(u == chunk_unit for u in question_units):
        score += 0.15
    
    # Factor 4: Magnitude reasonableness (rough heuristic)
    if 0 < number < 1e12:  # Reasonable for energy data
        score += 0.1
    elif number == 0 or number > 1e12:
        score -= 0.1  # Suspiciously small or large
    
    # Factor 5: Temporal relevance
    question_years = _extract_years_from_question(question)
    chunk_temporal = _extract_temporal(chunk.record.content_text)
    if question_years:
        for year in question_years:
            if str(year) in chunk_temporal:
                score += 0.1
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def _extract_domain_from_question(question: str) -> str:
    """Extract likely domain from question."""
    question_lower = question.lower()
    
    domains = {
        'emissions': r'\bemission|CO2|GHG|greenhouse|scope|carbon',
        'energy': r'\benergy|electricity|power|MWh|GWh|kWh|renewable',
        'water': r'\bwater|withdrawn|consumed|m³|liters|gallons',
        'waste': r'\bwaste|hazardous|recycl',
        'biodiversity': r'\bbiodiversity|species|habitat|protected',
    }
    
    for domain, pattern in domains.items():
        if re.search(pattern, question_lower):
            return domain
    
    return 'other'


def _extract_units_from_question(question: str) -> list[str]:
    """Extract mentioned units from question."""
    units = []
    text_lower = question.lower()
    
    unit_patterns = {
        'MWh': r'\bMWh\b|\bmwh\b',
        'GWh': r'\bGWh\b|\bgwh\b',
        'kWh': r'\bkWh\b|\bkwh\b',
        'tons': r'\bton[s]?\b',
        'm3': r'\bm[³3]\b',
        '%': r'%|percent',
    }
    
    for unit, pattern in unit_patterns.items():
        if re.search(pattern, text_lower):
            units.append(unit)
    
    return units


def _extract_years_from_question(question: str) -> list[str]:
    """Extract year references from question."""
    years = re.findall(r'\b(20\d{2})\b', question)
    return years


def extract_candidates(
    question: str,
    retrieved_chunks: list[RetrievedEvidence],
) -> list[CandidateNumber]:
    """Extract all numeric candidates from retrieved chunks.
    
    For each chunk:
    1. Find all numbers
    2. Extract metadata (domain, unit, temporal)
    3. Score by relevance to question intent
    4. Return ranked list (highest confidence first)
    """
    candidates = []
    
    for chunk_idx, chunk in enumerate(retrieved_chunks):
        if not chunk.record.content_text:
            continue
        
        numbers = _extract_numbers(chunk.record.content_text)
        
        for number in numbers:
            candidate = CandidateNumber(
                value=number,
                source_chunk=chunk,
                domain=chunk.record.domain if hasattr(chunk.record, 'domain') else 'other',
                unit=_extract_unit(chunk.record.content_text),
                temporal=_extract_temporal(chunk.record.content_text),
                confidence=_score_candidate_relevance(
                    question,
                    chunk,
                    number,
                    chunk_idx,
                ),
                source_text=chunk.record.content_text[:100],
                chunk_index=chunk_idx,
            )
            candidates.append(candidate)
    
    # Sort by confidence (descending)
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def _group_by_source_table(candidates: list[CandidateNumber]) -> list[list[CandidateNumber]]:
    """Group candidates by their source table/chunk."""
    groups = {}
    for candidate in candidates:
        chunk_id = id(candidate.source_chunk.record)
        if chunk_id not in groups:
            groups[chunk_id] = []
        groups[chunk_id].append(candidate)
    return list(groups.values())


def _group_by_temporal(candidates: list[CandidateNumber]) -> list[list[CandidateNumber]]:
    """Group candidates by temporal reference."""
    groups = {}
    for candidate in candidates:
        temporal = candidate.temporal or 'unspecified'
        if temporal not in groups:
            groups[temporal] = []
        groups[temporal].append(candidate)
    return list(groups.values())


def _group_by_unit(candidates: list[CandidateNumber]) -> list[list[CandidateNumber]]:
    """Group candidates by unit."""
    groups = {}
    for candidate in candidates:
        unit = candidate.unit or 'unknown'
        if unit not in groups:
            groups[unit] = []
        groups[unit].append(candidate)
    return list(groups.values())


def select_operands(
    question: str,
    intent: str,
    candidates: list[CandidateNumber],
) -> tuple[list[CandidateNumber], float]:
    """Select operands based on question intent.
    
    Rules:
    - Single-value questions: Select #1 candidate (highest confidence)
    - Multi-value questions (sum/average): Select all from same table/domain
    - Comparison (difference/change): Select best from each temporal group
    
    Returns:
        (selected_operands, overall_confidence)
    """
    
    if not candidates:
        return [], 0.0
    
    if intent in ("sum", "total", "combined", "aggregate"):
        # Multi-value: group by source table, select all from best group
        groups = _group_by_source_table(candidates)
        if groups:
            best_group = max(groups, key=lambda g: sum(c.confidence for c in g) / len(g))
            overall_conf = sum(c.confidence for c in best_group) / len(best_group)
            return best_group, overall_conf
        return candidates[:1], candidates[0].confidence
    
    elif intent in ("average", "mean", "median"):
        # Multi-value: select all numeric values of same unit from top table
        same_unit = [c for c in candidates if c.unit == candidates[0].unit]
        if len(same_unit) > 1:
            overall_conf = sum(c.confidence for c in same_unit) / len(same_unit)
            return same_unit, overall_conf
        return candidates[:1], candidates[0].confidence if candidates else 0.0
    
    elif intent in ("difference", "change", "increase", "decrease", "reduction", "growth"):
        # Two values: temporal groups (before/after) or categorical
        temporal_groups = _group_by_temporal(candidates)
        
        if len(temporal_groups) >= 2:
            # Select best from first two temporal groups
            before = max(temporal_groups[0], key=lambda c: c.confidence)
            after = max(temporal_groups[1], key=lambda c: c.confidence)
            overall_conf = (before.confidence + after.confidence) / 2
            return [before, after], overall_conf
        elif len(candidates) >= 2:
            # Fallback: select top 2
            overall_conf = (candidates[0].confidence + candidates[1].confidence) / 2
            return candidates[:2], overall_conf
        else:
            return candidates[:1], candidates[0].confidence if candidates else 0.0
    
    elif intent in ("percentage", "ratio", "share", "proportion"):
        # Percentage: typically single value that represents %
        pct_candidates = [c for c in candidates if '%' in c.unit or (0 <= c.value <= 100)]
        if pct_candidates:
            return pct_candidates[:1], pct_candidates[0].confidence
        return candidates[:1], candidates[0].confidence if candidates else 0.0
    
    else:  # Single value (extractive, comparison without change)
        return candidates[:1], candidates[0].confidence if candidates else 0.0


def should_use_grounded_operands(
    selected: list[CandidateNumber],
    confidence: float,
    threshold: float = 0.6,
) -> bool:
    """Determine if selected operands are trustworthy.
    
    If confidence < threshold:
    - Don't proceed with calculation
    - Return INSUFFICIENT_CONTEXT
    - Don't try to compensate with tools
    """
    return confidence >= threshold
