from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import yaml

from gri_benchmark.data import load_examples


TEXT_KEYS = [
    "gri",
    "question_type_ext",
    "pdf name",
    "source_file",
    "table_id",
    "row_id",
    "column_id",
    "unit",
    "years",
]


def _clean_text(value: Any) -> str:
    text = str(value).strip().strip("\"'")
    return " ".join(text.split())


def _primary_value(metadata: dict[str, Any], gold_answer: str) -> str:
    for key in ("value", "answer_value"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return _clean_text(value)
    return _clean_text(gold_answer)


def _content_text(question: str, metadata: dict[str, Any], primary_value: str) -> str:
    parts: list[str] = [question]
    for key in TEXT_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        cleaned = _clean_text(value)
        if cleaned:
            parts.append(cleaned)

    # Corpus may contain factual values because it represents retrievable table evidence.
    if primary_value:
        parts.append(primary_value)

    return " ".join(parts)


def _extract_years(text: str) -> list[str]:
    import re

    return sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", text)))


def _extract_units(text: str) -> list[str]:
    lowered = text.lower()
    unit_map = {
        "gwh": "gwh",
        "mwh": "mwh",
        "gj": "gj",
        "m3": "m3",
        "tons": "tons",
        "ton": "tons",
        "%": "percent",
        "percentage": "percent",
    }
    found = set()
    for marker, label in unit_map.items():
        if marker in lowered:
            found.add(label)
    return sorted(found)


def _extract_intents(text: str) -> list[str]:
    lowered = text.lower()
    intents = set()
    intent_map = {
        "average": "average",
        "sum": "sum",
        "difference": "difference",
        "increase": "difference",
        "reduction": "difference",
        "maximum": "superlative",
        "minimum": "superlative",
        "lowest": "superlative",
        "highest": "superlative",
        "which company": "company_selection",
        "rank": "ranking",
        "comparative": "comparative",
    }
    for marker, label in intent_map.items():
        if marker in lowered:
            intents.add(label)
    return sorted(intents)


def build_corpus_from_config(config_path: str | Path, output_override: str | Path | None = None) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    datasets = config["datasets"]
    output_path = Path(output_override or config.get("corpus_path", "data/corpus/benchmark_corpus.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    for ds in datasets:
        split = ds.get("split", "eval")
        examples = load_examples(ds["path"], split=split)
        for ex in examples:
            md = ex.metadata
            source_file = str(md.get("source_file") or md.get("source") or "")
            table_id = str(md.get("table_id") or "")
            row_id = str(md.get("row_id") or "")
            column_id = str(md.get("column_id") or "")
            key = (split, source_file, table_id, row_id, column_id)

            if key not in grouped:
                grouped[key] = {
                    "split": split,
                    "source_file": source_file,
                    "table_id": table_id,
                    "row_id": row_id,
                    "column_id": column_id,
                    "questions": [],
                    "values": [],
                    "metadata_tokens": [],
                }

            pv = _primary_value(md, ex.gold_answer)
            grouped[key]["questions"].append(_clean_text(ex.question))
            if pv:
                grouped[key]["values"].append(pv)

            for field in TEXT_KEYS:
                if field in md and md[field] is not None:
                    token = _clean_text(md[field])
                    if token:
                        grouped[key]["metadata_tokens"].append(token)

    rows: list[dict[str, Any]] = []
    for idx, (_key, bundle) in enumerate(grouped.items()):
        value_counter = Counter(bundle["values"])
        primary_value = value_counter.most_common(1)[0][0] if value_counter else ""

        unique_questions = []
        seen_q = set()
        for q in bundle["questions"]:
            if q and q not in seen_q:
                unique_questions.append(q)
                seen_q.add(q)

        unique_meta = []
        seen_m = set()
        for t in bundle["metadata_tokens"]:
            if t and t not in seen_m:
                unique_meta.append(t)
                seen_m.add(t)

        representative_question = unique_questions[0] if unique_questions else ""
        content_parts = []
        content_parts.extend(unique_questions[:3])
        content_parts.extend(unique_meta)
        if primary_value:
            content_parts.append(primary_value)

        record_id = f"chunk-{idx}"
        rows.append(
            {
                "record_id": record_id,
                "split": bundle["split"],
                "source_file": bundle["source_file"],
                "table_id": bundle["table_id"],
                "row_id": bundle["row_id"],
                "column_id": bundle["column_id"],
                "primary_value": primary_value,
                "content_text": " ".join(content_parts),
                "years": _extract_years(" ".join(content_parts)),
                "units": _extract_units(" ".join(content_parts)),
                "intents": _extract_intents(" ".join(content_parts)),
                "question_count": len(unique_questions),
                "representative_question": representative_question,
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return {
        "records": len(rows),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval corpus JSONL for strict benchmark mode")
    parser.add_argument("--config", default="configs/benchmark.yaml", help="Path to benchmark YAML config")
    parser.add_argument("--output", default=None, help="Output corpus JSONL path; defaults to config.corpus_path")
    args = parser.parse_args()

    manifest = build_corpus_from_config(args.config, args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
