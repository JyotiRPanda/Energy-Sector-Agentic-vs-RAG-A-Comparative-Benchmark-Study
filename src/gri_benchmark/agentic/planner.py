from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


QueryType = Literal["extractive", "relational", "quantitative", "multi_step"]


@dataclass(frozen=True)
class PlanStep:
    step: str
    rationale: str


def classify_query_type(question: str, split: str) -> QueryType:
    q = question.lower()
    s = split.lower()

    if "multi" in s or "multistep" in s:
        return "multi_step"
    if "rel" in s:
        return "relational"
    if "quant" in s:
        return "quantitative"

    if any(k in q for k in ("sum", "total", "difference", "percent", "average", "ratio")):
        return "quantitative"
    if any(k in q for k in ("compare", "which company", "higher", "lower", "between")):
        return "relational"
    if any(k in q for k in ("step", "then", "after", "before", "across tables")):
        return "multi_step"
    return "extractive"


def generate_lightweight_execution_plan(
    query_type: QueryType,
    *,
    use_calculation_tool: bool,
    use_verifier: bool,
) -> list[PlanStep]:
    steps: list[PlanStep] = [
        PlanStep("table_lookup", "Retrieve top candidate table rows with constraints"),
        PlanStep("rerank", "Rerank candidates for semantic relevance"),
        PlanStep("row_column_select", "Pick the best row/column cell to ground answer"),
    ]

    if use_calculation_tool and query_type in {"quantitative", "multi_step"}:
        steps.append(PlanStep("numeric_calculation", "Compute numeric expression when required"))

    steps.append(PlanStep("answer_synthesis", "Generate final grounded answer"))

    if use_verifier:
        steps.append(PlanStep("citation_verification", "Check answer is supported by chosen evidence"))

    return steps


def build_explicit_plan_payload(
    query_type: QueryType,
    *,
    use_calculation_tool: bool,
    use_verifier: bool,
) -> dict[str, object]:
    plan_steps = generate_lightweight_execution_plan(
        query_type,
        use_calculation_tool=use_calculation_tool,
        use_verifier=use_verifier,
    )

    required_tools = ["lookup", "rerank"]
    if use_calculation_tool and query_type in {"quantitative", "multi_step"}:
        required_tools.append("calculation")
    required_tools.append("synthesis")
    if use_verifier:
        required_tools.append("verification")

    return {
        "task_type": query_type,
        "required_tools": required_tools,
        "execution_order": [step.step for step in plan_steps],
        "steps": [step.__dict__ for step in plan_steps],
    }
