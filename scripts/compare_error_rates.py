from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict[str, dict[str, float]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(
    summary: dict[str, dict[str, float]],
    pipeline: str,
    metric: str,
    *,
    default: float | None = None,
) -> float | None:
    pipeline_metrics = summary.get(pipeline, {})
    value = pipeline_metrics.get(metric, default)
    if value is None:
        return None
    return float(value)


def _fmt_delta(delta: float) -> str:
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare wrong_year and wrong_unit rates between benchmark runs")
    parser.add_argument("--current", default="results/summary.json", help="Path to current summary JSON")
    parser.add_argument("--previous", default="results/summary.previous.json", help="Path to previous summary JSON")
    args = parser.parse_args()

    current_path = Path(args.current)
    previous_path = Path(args.previous)

    if not current_path.exists():
        raise SystemExit(f"Current summary file not found: {current_path}")

    current = _load_json(current_path)
    previous = _load_json(previous_path) if previous_path.exists() else None

    tracked = ["error_rate.wrong_year", "error_rate.wrong_unit"]
    pipelines = sorted(current.keys())

    print("Error-rate delta report")
    print(f"Current: {current_path}")
    if previous is None:
        print(f"Previous: {previous_path} (not found; showing current values only)")
    else:
        print(f"Previous: {previous_path}")

    for pipeline in pipelines:
        print(f"\nPipeline: {pipeline}")
        for metric in tracked:
            curr = _metric_value(current, pipeline, metric, default=0.0)
            if curr is None:
                print(f"  {metric}: n/a")
                continue

            if previous is None:
                print(f"  {metric}: {curr:.6f}")
                continue

            prev = _metric_value(previous, pipeline, metric, default=0.0)
            if prev is None:
                prev = 0.0

            delta = curr - prev
            print(
                f"  {metric}: current={curr:.6f} | previous={prev:.6f} | delta={_fmt_delta(delta)}"
            )


if __name__ == "__main__":
    main()
