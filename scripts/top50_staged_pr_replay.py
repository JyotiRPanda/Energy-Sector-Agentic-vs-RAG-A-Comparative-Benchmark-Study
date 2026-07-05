from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, median

import yaml

from gri_benchmark.data import load_examples
from gri_benchmark.evidence import SimpleEvidenceRetriever
from gri_benchmark.live_clients import maybe_create_live_client
from gri_benchmark.pipelines.agentic_pipeline import AgenticMultiToolPipeline


def parse_first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+", str(text).replace(",", ""))
    if not match:
        return None
    return float(match.group(0))


def nre_old(gold: str, pred: str) -> float | None:
    g = parse_first_number(gold)
    p = parse_first_number(pred)
    if g is None or p is None:
        return None
    denom = abs(g) if g != 0 else 1.0
    return abs(p - g) / denom


def nre_new(gold: str, pred: str) -> float | None:
    g = parse_first_number(gold)
    p = parse_first_number(pred)
    if g is None or p is None:
        return None
    denom = abs(g) + abs(p)
    if denom < 1e-9:
        return 0.0
    return (2.0 * abs(p - g)) / denom


def agg(values: list[float | None]) -> dict[str, float | int | None]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": len(filtered),
        "mean": mean(filtered),
        "median": median(filtered),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/benchmark_full.yaml").read_text())

    examples = []
    for ds in cfg["datasets"]:
        examples.extend(load_examples(ds["path"], split=ds.get("split", "eval")))
    ex_by_id = {e.question_id: e for e in examples}

    baseline_preds = json.loads((root / "results/full/agentic_multi_tool_predictions.json").read_text())
    baseline_by_id = {p["question_id"]: p for p in baseline_preds}

    baseline_rows: list[tuple[float, str]] = []
    for qid, pred in baseline_by_id.items():
        score = nre_old(ex_by_id[qid].gold_answer, str(pred.get("answer", "")))
        if score is not None:
            baseline_rows.append((score, qid))
    baseline_rows.sort(reverse=True)
    top50_ids = [qid for _, qid in baseline_rows[:50]]

    retriever = None
    if cfg.get("strict_mode", False):
        retriever = SimpleEvidenceRetriever.from_jsonl(cfg["corpus_path"])

    live_client = maybe_create_live_client(force=True)
    if live_client is None:
        raise RuntimeError("Live client unavailable for top-50 staged replay.")

    pipeline = AgenticMultiToolPipeline(
        strict_mode=bool(cfg.get("strict_mode", False)),
        retriever=retriever,
        live_client=live_client,
    )

    after_preds: dict[str, dict] = {}
    for idx, qid in enumerate(top50_ids, start=1):
        pred = pipeline.answer(ex_by_id[qid])
        after_preds[qid] = {
            "answer": pred.answer,
            "metadata": pred.metadata,
        }
        if idx % 10 == 0:
            print(f"[top50-staged-replay] progress={idx}/50", flush=True)

    per_id = []
    for qid in top50_ids:
        ex = ex_by_id[qid]
        before_answer = str(baseline_by_id[qid].get("answer", ""))
        after_answer = str(after_preds[qid].get("answer", ""))

        before_old = nre_old(ex.gold_answer, before_answer)
        after_old = nre_old(ex.gold_answer, after_answer)
        before_new = nre_new(ex.gold_answer, before_answer)
        after_new = nre_new(ex.gold_answer, after_answer)

        md = after_preds[qid].get("metadata", {})
        sel_meta = md.get("selection_meta", {}) if isinstance(md, dict) else {}

        per_id.append(
            {
                "question_id": qid,
                "question": ex.question,
                "gold_answer": ex.gold_answer,
                "before_answer": before_answer,
                "after_answer": after_answer,
                "before_nre_old": before_old,
                "after_nre_old": after_old,
                "before_nre_new": before_new,
                "after_nre_new": after_new,
                "final_answer_source_after": md.get("final_answer_source"),
                "ratio_scale_guard_after": sel_meta.get("ratio_scale_guard"),
                "expected_table_constraint_after": sel_meta.get("expected_table_constraint"),
            }
        )

    summary = {
        "baseline_old_metric": agg([r["before_nre_old"] for r in per_id]),
        "stageA_pipeline_old_metric": agg([r["after_nre_old"] for r in per_id]),
        "baseline_new_metric_recomputed": agg([r["before_nre_new"] for r in per_id]),
        "stageB_pipeline_plus_new_metric": agg([r["after_nre_new"] for r in per_id]),
        "stageA_improved_count_old_metric": sum(
            1
            for r in per_id
            if r["before_nre_old"] is not None
            and r["after_nre_old"] is not None
            and r["after_nre_old"] < r["before_nre_old"]
        ),
        "stageA_worse_count_old_metric": sum(
            1
            for r in per_id
            if r["before_nre_old"] is not None
            and r["after_nre_old"] is not None
            and r["after_nre_old"] > r["before_nre_old"]
        ),
    }

    report = {
        "top50_ids": top50_ids,
        "summary": summary,
        "per_id": per_id,
    }

    out_json = root / "results/analysis/pr_stage_top50_before_after.json"
    out_md = root / "results/analysis/pr_stage_top50_before_after.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True))

    ranked = sorted(
        [
            r
            for r in per_id
            if r["before_nre_old"] is not None and r["after_nre_old"] is not None
        ],
        key=lambda r: (r["before_nre_old"] - r["after_nre_old"]),
        reverse=True,
    )

    lines = [
        "# Staged PR Top-50 Before/After",
        "",
        "## Stage Summary",
        f"- Baseline (old metric) mean NRE: {summary['baseline_old_metric']['mean']}",
        f"- Stage A (pipeline changes, old metric) mean NRE: {summary['stageA_pipeline_old_metric']['mean']}",
        f"- Baseline recomputed (new metric) mean NRE: {summary['baseline_new_metric_recomputed']['mean']}",
        f"- Stage B (pipeline + new metric) mean NRE: {summary['stageB_pipeline_plus_new_metric']['mean']}",
        f"- Stage A improved IDs: {summary['stageA_improved_count_old_metric']}",
        f"- Stage A worse IDs: {summary['stageA_worse_count_old_metric']}",
        "",
        "## Top 15 Largest Stage A Improvements (old metric)",
    ]

    for row in ranked[:15]:
        delta = float(row["before_nre_old"] - row["after_nre_old"])
        lines.append(
            "- "
            + f"{row['question_id']}: delta={delta:.4f}, before={row['before_nre_old']:.4f}, "
            + f"after={row['after_nre_old']:.4f}, source_after={row['final_answer_source_after']}"
        )

    out_md.write_text("\n".join(lines))

    print(f"[top50-staged-replay] wrote {out_json}")
    print(f"[top50-staged-replay] wrote {out_md}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
