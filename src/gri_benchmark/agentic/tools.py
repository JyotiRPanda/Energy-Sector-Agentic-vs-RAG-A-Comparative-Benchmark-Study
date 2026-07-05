from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable

from gri_benchmark.evidence import RetrievedEvidence


@dataclass
class ToolInvocation:
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: dict[str, Any]
    success: bool
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_output_preview(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return {"kind": "list", "size": len(value)}
    if isinstance(value, dict):
        return {"kind": "dict", "keys": list(value.keys())[:10]}
    return {"kind": "scalar", "value": str(value)[:120]}


def _safe_input_preview(payload: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = f"list(len={len(value)})"
        elif isinstance(value, dict):
            safe[key] = f"dict(keys={list(value.keys())[:8]})"
        else:
            safe[key] = f"{type(value).__name__}"
    return safe


def invoke_tool(
    *,
    tool_name: str,
    fn: Callable[..., Any],
    invocation_log: list[ToolInvocation],
    **kwargs: Any,
) -> Any:
    start = perf_counter()
    try:
        out = fn(**kwargs)
        invocation_log.append(
            ToolInvocation(
                tool_name=tool_name,
                tool_input=_safe_input_preview(kwargs),
                tool_output=_safe_output_preview(out),
                success=True,
                latency_ms=(perf_counter() - start) * 1000,
            )
        )
        return out
    except Exception as exc:  # pragma: no cover - defensive log path
        invocation_log.append(
            ToolInvocation(
                tool_name=tool_name,
                tool_input=_safe_input_preview(kwargs),
                tool_output={},
                success=False,
                latency_ms=(perf_counter() - start) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        raise


def table_lookup_tool(
    *,
    retriever: Any,
    question: str,
    split: str,
    source_file: str | None,
    top_k: int,
    use_constraints: bool,
) -> list[RetrievedEvidence]:
    if retriever is None:
        return []
    return retriever.search(
        question,
        split=split,
        source_file=source_file,
        top_k=top_k,
        use_constraints=use_constraints,
    )


def row_column_selector_tool(*, candidates: list[RetrievedEvidence]) -> RetrievedEvidence | None:
    return candidates[0] if candidates else None


def numeric_calculation_tool(*, question: str, value: str) -> str:
    # Lightweight calculation placeholder for quantitative/multi-step prompts.
    # If no explicit operation is recognized, pass through the selected value.
    lower_q = question.lower()
    if "percent" in lower_q and value:
        return value
    if any(k in lower_q for k in ("sum", "total", "difference", "average", "ratio")):
        return value
    return value


def citation_verifier_tool(*, answer: str, candidates: list[RetrievedEvidence]) -> bool:
    if not answer or answer == "INSUFFICIENT_CONTEXT" or not candidates:
        return False

    top = candidates[0].record
    primary = str(top.primary_value or "").strip()
    content = str(top.content_text or "").lower()
    ans = str(answer).strip().lower()

    if primary and ans == primary.lower():
        return True

    n_ans = re.search(r"[-+]?\d*\.?\d+", ans.replace(",", ""))
    n_top = re.search(r"[-+]?\d*\.?\d+", primary.replace(",", "")) if primary else None
    if n_ans and n_top:
        a = float(n_ans.group(0))
        b = float(n_top.group(0))
        denom = abs(b) if b != 0 else 1.0
        if abs(a - b) / denom <= 1e-3:
            return True

    return ans in content if ans else False


def answer_synthesis_tool(*, selected_value: str, generated_answer: str | None = None) -> str:
    if generated_answer and str(generated_answer).strip():
        return str(generated_answer).strip()
    if selected_value and str(selected_value).strip():
        return str(selected_value).strip()
    return "INSUFFICIENT_CONTEXT"
