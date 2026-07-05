from __future__ import annotations

import re
import random
from statistics import mean

from gri_benchmark.types import BenchmarkExample, Prediction


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _numeric_sequence(text: str) -> list[float]:
    nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    out: list[float] = []
    for n in nums:
        try:
            out.append(float(n))
        except ValueError:
            continue
    return out


def _non_numeric_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[-+]?\d*\.?\d+", " ", lowered)
    lowered = re.sub(r"[^a-z]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _norm_optional(value: object) -> str:
    text = str(value or "").strip()
    return _normalize(text) if text else ""


def exact_match(gold: str, pred: str) -> float:
    if _normalize(gold) == _normalize(pred):
        return 1.0

    # Numeric-sequence equivalence for formatting-only differences
    # (e.g., "37789" vs "37789.0", "288, 375" vs "288.0, 375.0").
    g_nums = _numeric_sequence(gold)
    p_nums = _numeric_sequence(pred)
    if g_nums and p_nums and len(g_nums) == len(p_nums):
        if _non_numeric_text(gold) == _non_numeric_text(pred):
            if all(abs(g - p) <= 1e-9 for g, p in zip(g_nums, p_nums)):
                return 1.0

    return 0.0


def numeric_tolerance_match(gold: str, pred: str, tolerance_pct: float = 5.0) -> float:
    """Check if predicted value matches gold value within numeric tolerance.
    
    Args:
        gold: Gold/expected value
        pred: Predicted value
        tolerance_pct: Tolerance as percentage (default 5%)
    
    Returns:
        1.0 if values match within tolerance, 0.0 otherwise
    """
    g = _parse_first_number(gold)
    p = _parse_first_number(pred)
    
    if g is None or p is None:
        return 0.0
    
    # Exact numeric match (within floating-point precision)
    if abs(p - g) < 1e-9:
        return 1.0
    
    # Tolerance-based match
    denom = abs(g) if g != 0 else 1.0
    relative_error = abs(p - g) / denom
    tolerance = tolerance_pct / 100.0
    
    # Add small epsilon for floating-point comparison
    if relative_error <= (tolerance + 1e-9):
        return 1.0
    
    return 0.0


def _parse_first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if not match:
        return None
    return float(match.group(0))


def numeric_relative_error(gold: str, pred: str) -> float | None:
    g = _parse_first_number(gold)
    p = _parse_first_number(pred)
    if g is None or p is None:
        return None
    # Symmetric relative error avoids denominator explosions for near-zero gold values.
    denom = abs(g) + abs(p)
    if denom < 1e-9:
        return 0.0
    return (2.0 * abs(p - g)) / denom


def _expected_citation_fields(example: BenchmarkExample) -> dict[str, str]:
    md = example.metadata
    return {
        "source_file": _norm_optional(md.get("source_file") or md.get("source")),
        "table_id": _norm_optional(md.get("table_id")),
        "row_id": _norm_optional(md.get("row_id")),
        "column_id": _norm_optional(md.get("column_id")),
    }


def _citation_matches_expected(citation: object, expected: dict[str, str]) -> bool:
    src_expected = expected.get("source_file", "")
    table_expected = expected.get("table_id", "")
    row_expected = expected.get("row_id", "")
    col_expected = expected.get("column_id", "")

    src_obs = _norm_optional(getattr(citation, "source_file", ""))
    table_obs = _norm_optional(getattr(citation, "table_id", ""))
    row_obs = _norm_optional(getattr(citation, "row_id", ""))
    col_obs = _norm_optional(getattr(citation, "column_id", ""))

    # Source/table are the strongest anchors; row/col are optional refinements.
    source_ok = (not src_expected) or (src_expected == src_obs)
    table_ok = (not table_expected) or (table_expected == table_obs)
    row_ok = (not row_expected) or (row_expected == row_obs)
    col_ok = (not col_expected) or (col_expected == col_obs)
    return source_ok and table_ok and row_ok and col_ok


def citation_validity_score(example: BenchmarkExample, prediction: Prediction) -> float:
    if not prediction.citations:
        return 0.0

    expected = _expected_citation_fields(example)
    valid = sum(1 for c in prediction.citations if _citation_matches_expected(c, expected))
    return valid / len(prediction.citations)


def _answer_supported_by_retrieval(prediction: Prediction) -> bool:
    pred_text = _normalize(prediction.answer)
    pred_num = _parse_first_number(prediction.answer)
    hits = prediction.metadata.get("retrieval_hits", [])
    if not isinstance(hits, list) or not hits:
        return False

    for hit in hits:
        if not isinstance(hit, dict):
            continue
        primary_value = str(hit.get("primary_value", "")).strip()
        content_text = str(hit.get("content_text", "")).strip()

        if primary_value and _normalize(primary_value) == pred_text:
            return True

        hit_num = _parse_first_number(primary_value) if primary_value else None
        if pred_num is not None and hit_num is not None:
            denom = abs(hit_num) if hit_num != 0 else 1.0
            if abs(pred_num - hit_num) / denom <= 1e-3:
                return True

        if content_text and pred_text and pred_text in _normalize(content_text):
            return True

    return False


def support_level(example: BenchmarkExample, prediction: Prediction) -> str:
    citation_ok = citation_validity_score(example, prediction) >= 0.5
    evidence_ok = _answer_supported_by_retrieval(prediction)

    if citation_ok and evidence_ok:
        return "supported"
    if citation_ok or evidence_ok:
        return "partially_supported"
    return "unsupported"


def faithfulness_score(example: BenchmarkExample, prediction: Prediction) -> float:
    level = support_level(example, prediction)
    if level == "supported":
        return 1.0
    if level == "partially_supported":
        return 0.5
    return 0.0


def citation_precision(example: BenchmarkExample, prediction: Prediction) -> float | None:
    return citation_validity_score(example, prediction)


def citation_recall(prediction: Prediction, example: BenchmarkExample) -> float | None:
    expected = float(example.metadata.get("expected_citation_count", 1.0))
    if expected <= 0:
        return 1.0
    if expected == 1.0:
        # For the common single-citation case, recall equals whether at least one valid citation exists.
        return 1.0 if citation_validity_score(example, prediction) > 0 else 0.0
    return min(1.0, len(prediction.citations) / expected)


def transparency_score(prediction: Prediction) -> float:
    has_trace = float(bool(prediction.trace_steps))
    has_citations = float(bool(prediction.citations))
    return (has_trace + has_citations) / 2.0


def _bootstrap_ci(values: list[float], *, n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])

    rng = random.Random(42)
    n = len(values)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo_idx = int((alpha / 2) * (n_bootstrap - 1))
    hi_idx = int((1 - alpha / 2) * (n_bootstrap - 1))
    return (means[lo_idx], means[hi_idx])


def aggregate_metrics(examples: list[BenchmarkExample], predictions: list[Prediction]) -> dict[str, float]:
    ex_by_id = {e.question_id: e for e in examples}
    em_list: list[float] = []
    nre_list: list[float] = []
    cp_list: list[float] = []
    cr_list: list[float] = []
    faith_list: list[float] = []
    trans_list: list[float] = []
    lat_list: list[float] = []
    prompt_tokens_list: list[float] = []
    completion_tokens_list: list[float] = []
    embedding_tokens_list: list[float] = []
    cost_list: list[float] = []
    tool_calls_list: list[float] = []
    planning_ms_list: list[float] = []
    retrieval_ms_list: list[float] = []
    tool_exec_ms_list: list[float] = []
    synthesis_ms_list: list[float] = []

    for pred in predictions:
        ex = ex_by_id[pred.question_id]
        em_list.append(exact_match(ex.gold_answer, pred.answer))

        nre = numeric_relative_error(ex.gold_answer, pred.answer)
        if nre is not None:
            nre_list.append(nre)

        cp = citation_precision(ex, pred)
        if cp is not None:
            cp_list.append(cp)

        cr = citation_recall(pred, ex)
        if cr is not None:
            cr_list.append(cr)

        faith_list.append(faithfulness_score(ex, pred))
        trans_list.append(transparency_score(pred))
        lat_list.append(pred.latency_ms)

        token_usage = pred.metadata.get("token_usage", {}) if isinstance(pred.metadata, dict) else {}
        prompt_tokens_list.append(float(token_usage.get("prompt_tokens", 0) or 0))
        completion_tokens_list.append(float(token_usage.get("completion_tokens", 0) or 0))
        embedding_tokens_list.append(float(token_usage.get("embedding_tokens", 0) or 0))
        cost_list.append(float(pred.metadata.get("cost_usd", 0.0) or 0.0))
        tool_calls_list.append(float(pred.metadata.get("tool_calls", 0) or 0))

        lb = pred.metadata.get("latency_breakdown_ms", {}) if isinstance(pred.metadata, dict) else {}
        planning_ms_list.append(float(lb.get("planning", 0.0) or 0.0))
        retrieval_ms_list.append(float(lb.get("retrieval", 0.0) or 0.0))
        tool_exec_ms_list.append(float(lb.get("tool_execution", 0.0) or 0.0))
        synthesis_ms_list.append(float(lb.get("synthesis", 0.0) or 0.0))

    em_ci = _bootstrap_ci(em_list)
    cp_ci = _bootstrap_ci(cp_list)
    faith_ci = _bootstrap_ci(faith_list)

    return {
        "n_samples": float(len(predictions)),
        "exact_match": mean(em_list) if em_list else 0.0,
        "exact_match_ci_low": em_ci[0],
        "exact_match_ci_high": em_ci[1],
        "numeric_relative_error": mean(nre_list) if nre_list else 0.0,
        "citation_precision": mean(cp_list) if cp_list else 0.0,
        "citation_precision_ci_low": cp_ci[0],
        "citation_precision_ci_high": cp_ci[1],
        "citation_recall": mean(cr_list) if cr_list else 0.0,
        "faithfulness": mean(faith_list) if faith_list else 0.0,
        "faithfulness_ci_low": faith_ci[0],
        "faithfulness_ci_high": faith_ci[1],
        "transparency": mean(trans_list) if trans_list else 0.0,
        "latency_ms": mean(lat_list) if lat_list else 0.0,
        "avg_prompt_tokens": mean(prompt_tokens_list) if prompt_tokens_list else 0.0,
        "avg_completion_tokens": mean(completion_tokens_list) if completion_tokens_list else 0.0,
        "avg_embedding_tokens": mean(embedding_tokens_list) if embedding_tokens_list else 0.0,
        "avg_tool_calls": mean(tool_calls_list) if tool_calls_list else 0.0,
        "avg_cost_usd_per_sample": mean(cost_list) if cost_list else 0.0,
        "total_cost_usd": sum(cost_list) if cost_list else 0.0,
        "latency_planning_ms": mean(planning_ms_list) if planning_ms_list else 0.0,
        "latency_retrieval_ms": mean(retrieval_ms_list) if retrieval_ms_list else 0.0,
        "latency_tool_execution_ms": mean(tool_exec_ms_list) if tool_exec_ms_list else 0.0,
        "latency_synthesis_ms": mean(synthesis_ms_list) if synthesis_ms_list else 0.0,
    }
