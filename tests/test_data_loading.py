from __future__ import annotations

from gri_benchmark.data import load_examples


def test_load_examples_uses_value_column_as_gold_answer(tmp_path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "question,value,pdf name,table nbr,row,column\n"
        "What is total?,516.2,['doc.pdf'],[0],31,3\n",
        encoding="utf-8",
    )

    examples = load_examples(csv_path, split="single_table_extractive")

    assert len(examples) == 1
    ex = examples[0]
    assert ex.gold_answer == "516.2"
    assert ex.metadata["source_file"] == "doc.pdf"
    assert ex.metadata["table_id"] == "0"
    assert ex.metadata["row_id"] == "31"
    assert ex.metadata["column_id"] == "3"
