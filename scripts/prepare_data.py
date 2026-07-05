from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def _copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy selected GRI-QA benchmark CSV files into this repository")
    parser.add_argument("--source-root", required=True, help="Path to original gri_qa-main/dataset")
    parser.add_argument("--target-root", default="data/benchmark", help="Path to local benchmark data folder")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    one_table_files = [
        "one-table/gri-qa_extra.csv",
        "one-table/gri-qa_rel.csv",
        "one-table/gri-qa_quant.csv",
        "one-table/gri-qa_multistep.csv",
    ]
    multi_table_files = [
        "multi-table/gri-qa_multitable2-rel.csv",
        "multi-table/gri-qa_multitable2-quant.csv",
        "multi-table/gri-qa_multitable2-multistep.csv",
    ]

    copied = []
    for rel_path in one_table_files + multi_table_files:
        src = source_root / rel_path
        dst = target_root / rel_path
        if _copy_if_exists(src, dst):
            copied.append(rel_path)

    manifest = target_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path"])
        for rel_path in copied:
            writer.writerow([rel_path])

    print(f"Copied {len(copied)} files into {target_root}")
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
