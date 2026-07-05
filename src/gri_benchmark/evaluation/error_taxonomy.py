from __future__ import annotations

from collections import Counter
import re

from gri_benchmark.evaluation.metrics import citation_validity_score, numeric_relative_error, support_level
from gri_benchmark.types import BenchmarkExample, Prediction


def _extract_years(text: str) -> set[str]:
    return set(re.findall(r"\b(?:19|20)\d{2}\b", text))


def _extract_units(text: str) -> set[str]:
    """Extract and normalize energy/mass units from text.
    
    Normalizes variants:
    - GWh, MWh, GJ (energy)
    - tons, ton, t, T (mass)
    - m3, m^3, m³ (volume)
    - %, percent, percentage (ratio)
    """
    lowered = text.lower()
    units = set()
    padded = f" {lowered} "
    
    # Energy units
    if "gwh" in lowered:
        units.add("gwh")
    if "mwh" in lowered:
        units.add("mwh")
    if "gj" in lowered:
        units.add("gj")
    
    # Mass/weight units (normalize all variants to "tons")
    if any(m in lowered for m in ["tons", "ton"]):
        units.add("tons")
    if " t " in padded or "-t-" in lowered or "t " in lowered:
        units.add("tons")
    
    # Volume units (normalize to "m3")
    if any(m in lowered for m in ["m3", "m^3", "m³"]):
        units.add("m3")
    
    # Percentage/ratio units
    if "%" in lowered or "percent" in lowered:
        units.add("percent")
    
    return units


def _top_retrieval_hit(prediction: Prediction) -> dict | None:
    hits = prediction.metadata.get("retrieval_hits", [])
    if not isinstance(hits, list) or not hits:
        return None
    top = hits[0]
    if not isinstance(top, dict):
        return None
    return top


def classify_errors(example: BenchmarkExample, prediction: Prediction) -> list[str]:
    labels: list[str] = []

    if support_level(example, prediction) == "unsupported":
        labels.append("unsupported_claim")

    if citation_validity_score(example, prediction) < 0.5:
        labels.append("miscitation")

    nre = numeric_relative_error(example.gold_answer, prediction.answer)
    if nre is not None and nre > 0.05:
        labels.append("incorrect_quantitative_operation")

    tool_failure = bool(prediction.metadata.get("tool_failure", False))
    any_tool_error = any(step.get("status") == "error" for step in prediction.trace_steps)
    if tool_failure or any_tool_error:
        labels.append("tool_reasoning_failure")

    top_hit = _top_retrieval_hit(prediction)
    if top_hit is not None:
        expected_table = str(example.metadata.get("table_id", "")).strip()
        observed_table = str(top_hit.get("table_id", "")).strip()
        if expected_table and observed_table and expected_table != observed_table:
            labels.append("wrong_table")

        expected_years = _extract_years(example.question)
        observed_years = set(str(y) for y in top_hit.get("years", []) if str(y).strip())
        if expected_years and observed_years and not (expected_years & observed_years):
            labels.append("wrong_year")

        expected_units = _extract_units(example.question)
        observed_units = set(str(u) for u in top_hit.get("units", []) if str(u).strip())
        if expected_units and observed_units and not (expected_units & observed_units):
            labels.append("wrong_unit")

    return labels


def summarize_errors(examples: list[BenchmarkExample], predictions: list[Prediction]) -> dict[str, float]:
    ex_by_id = {e.question_id: e for e in examples}
    counter: Counter[str] = Counter()

    for pred in predictions:
        ex = ex_by_id[pred.question_id]
        counter.update(classify_errors(ex, pred))

    n = max(len(predictions), 1)
    return {key: value / n for key, value in sorted(counter.items())}
