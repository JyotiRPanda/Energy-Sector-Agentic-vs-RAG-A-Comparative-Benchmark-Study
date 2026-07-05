from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from gri_benchmark.types import BenchmarkExample


def _first_present(row: pd.Series, keys: list[str], default: str = "") -> str:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return str(row[key]).strip()
    return default


def _safe_literal(value: object) -> object:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value


def _to_citation_value(value: object) -> str | None:
    parsed = _safe_literal(value)
    if parsed is None:
        return None
    if isinstance(parsed, list):
        if not parsed:
            return None
        return str(parsed[0])
    text = str(parsed).strip()
    return text or None


def load_examples(csv_path: str | Path, split: str) -> list[BenchmarkExample]:
    frame = pd.read_csv(csv_path)
    examples: list[BenchmarkExample] = []

    for idx, row in frame.iterrows():
        question = _first_present(row, ["question", "query", "input"])
        gold_answer = _first_present(
            row,
            ["answer", "gold_answer", "label", "output", "value", "answer_value"],
        )
        question_id = _first_present(row, ["question_id", "id"], default=f"{split}-{idx}")

        metadata = {
            "row_index": int(idx),
            "source": str(csv_path),
            "source_file": _to_citation_value(row.get("pdf name", csv_path)),
            "table_id": _to_citation_value(row.get("table nbr")),
            "row_id": _to_citation_value(row.get("row", row.get("row indices"))),
            "column_id": _to_citation_value(row.get("column", row.get("col indices"))),
        }

        for k, v in row.items():
            if k not in {"question", "query", "input", "answer", "gold_answer", "label", "output"} and pd.notna(v):
                metadata[k] = v

        examples.append(
            BenchmarkExample(
                question_id=question_id,
                question=question,
                gold_answer=gold_answer,
                split=split,
                metadata=metadata,
            )
        )

    return examples
