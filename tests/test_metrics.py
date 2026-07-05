from gri_benchmark.evaluation.error_taxonomy import classify_errors
from gri_benchmark.evaluation.metrics import exact_match, numeric_relative_error
from gri_benchmark.types import BenchmarkExample, Prediction


def test_exact_match_normalized() -> None:
    assert exact_match(" 12.0 ", "12.0") == 1.0


def test_numeric_relative_error() -> None:
    err = numeric_relative_error("100", "110")
    assert err is not None
    assert round(err, 3) == 0.1


def test_error_classification() -> None:
    example = BenchmarkExample(
        question_id="q1",
        question="What is the total?",
        gold_answer="100",
        split="eval",
    )
    prediction = Prediction(
        question_id="q1",
        pipeline_name="x",
        answer="120",
        latency_ms=10.0,
        metadata={"support_score": 0.0, "citation_validity": 0.0, "tool_failure": True},
    )
    labels = classify_errors(example, prediction)
    assert "unsupported_claim" in labels
    assert "miscitation" in labels
    assert "incorrect_quantitative_operation" in labels
    assert "tool_reasoning_failure" in labels


def test_retrieval_error_labels() -> None:
    example = BenchmarkExample(
        question_id="q2",
        question="What was total energy in GWh in 2023?",
        gold_answer="100",
        split="eval",
        metadata={"table_id": "0"},
    )
    prediction = Prediction(
        question_id="q2",
        pipeline_name="x",
        answer="100",
        latency_ms=5.0,
        metadata={
            "support_score": 0.9,
            "citation_validity": 0.9,
            "retrieval_hits": [
                {
                    "record_id": "chunk-10",
                    "table_id": "2",
                    "years": ["2022"],
                    "units": ["mwh"],
                }
            ],
        },
    )

    labels = classify_errors(example, prediction)
    assert "wrong_table" in labels
    assert "wrong_year" in labels
    assert "wrong_unit" in labels


def test_retrieval_error_labels_with_constraint_metadata() -> None:
    """Test error classification with explicit constraint satisfaction metadata."""
    example = BenchmarkExample(
        question_id="q3",
        question="What was total energy in GWh in 2023?",
        gold_answer="500",
        split="eval",
        metadata={"table_id": "1"},
    )
    # Simulate retrieval result with year mismatch (2022 instead of 2023)
    prediction = Prediction(
        question_id="q3",
        pipeline_name="x",
        answer="400",  # 20% error to exceed 0.05 threshold
        latency_ms=3.0,
        metadata={
            "support_score": 0.8,
            "citation_validity": 0.8,
            "retrieval_hits": [
                {
                    "record_id": "energy_2022",
                    "table_id": "1",
                    "years": ["2022"],
                    "units": ["gwh"],
                    "score_breakdown": {
                        "lexical": 0.85,
                        "intent_match": 0.6,
                        "year_constraint_satisfied": False,
                        "unit_constraint_satisfied": True,
                        "from_fallback": True,
                        "fallback_reason": "no_year_match",
                        "expected_years": ["2023"],
                        "expected_units": ["gwh"],
                        "top_hit_years": ["2022"],
                        "top_hit_units": ["gwh"],
                    },
                }
            ],
        },
    )
    
    labels = classify_errors(example, prediction)
    # Should have wrong_year error due to year mismatch (2022 vs 2023)
    assert "wrong_year" in labels
    # Should NOT have wrong_unit error (both gwh)
    assert "wrong_unit" not in labels
    # Should have incorrect_quantitative_operation (400 vs 500, 20% error)
    assert "incorrect_quantitative_operation" in labels


def test_retrieval_constraint_satisfaction_in_score_breakdown() -> None:
    """Test that score_breakdown includes constraint satisfaction flags."""
    example = BenchmarkExample(
        question_id="q4",
        question="Total energy production in MWh for 2023",
        gold_answer="1000000",
        split="eval",
    )
    prediction = Prediction(
        question_id="q4",
        pipeline_name="x",
        answer="1000000",
        latency_ms=2.5,
        metadata={
            "support_score": 1.0,
            "citation_validity": 1.0,
            "retrieval_hits": [
                {
                    "record_id": "perfect_match",
                    "table_id": "3",
                    "years": ["2023"],
                    "units": ["mwh"],
                    "score_breakdown": {
                        "lexical": 0.95,
                        "intent_match": 0.8,
                        "year_constraint_satisfied": True,
                        "unit_constraint_satisfied": True,
                        "from_fallback": False,
                        "fallback_reason": None,
                        "expected_years": ["2023"],
                        "expected_units": ["mwh"],
                        "top_hit_years": ["2023"],
                        "top_hit_units": ["mwh"],
                    },
                }
            ],
        },
    )
    
    labels = classify_errors(example, prediction)
    # Perfect match: no errors should be flagged
    assert "wrong_year" not in labels
    assert "wrong_unit" not in labels
    assert "wrong_table" not in labels
