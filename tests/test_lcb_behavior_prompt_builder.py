from __future__ import annotations

import importlib.util
from pathlib import Path


def load_prompt_builder_module():
    script = Path(__file__).parents[1] / "scripts" / "build_lcb_behavior_test_prompts.py"
    spec = importlib.util.spec_from_file_location("build_lcb_behavior_test_prompts", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_signal_features_counts_fragility() -> None:
    builder = load_prompt_builder_module()

    features = builder.public_signal_features(
        {"question_id": "x", "public_scores": [1.0, 0.5, 1.0, 0.0]}
    )

    assert features["best_public_score"] == 1.0
    assert features["best_public_tie_count"] == 2
    assert features["public_pass_count"] == 2
    assert features["partial_public_score_count"] == 1
    assert features["nonzero_public_score_count"] == 3
    assert features["public_score_sum"] == 2.5


def test_prompt_public_problem_parses_parquet_row_shape() -> None:
    builder = load_prompt_builder_module()

    problem = builder.PromptPublicProblem(
        {
            "question_id": "abc001_a",
            "question_title": "Title",
            "question_content": "Statement",
            "difficulty": "easy",
            "platform": "atcoder",
            "metadata": '{"func_name": "solve"}',
            "public_test_cases": '[{"input": "1\\n", "output": "2\\n"}]',
        }
    )

    assert problem.question_id == "abc001_a"
    assert problem.metadata == {"func_name": "solve"}
    assert problem.public_test_cases[0].input == "1\n"
    assert problem.public_test_cases[0].output == "2\n"


def test_select_target_records_keeps_default_input_order() -> None:
    builder = load_prompt_builder_module()
    records = [
        {"question_id": "all-pass", "public_scores": [1.0, 1.0, 1.0, 1.0]},
        {"question_id": "fragile", "public_scores": [1.0, 0.5, 1.0, 0.0]},
        {"question_id": "single-pass", "public_scores": [1.0, 0.5, 0.0, 0.0]},
        {"question_id": "three-pass", "public_scores": [1.0, 1.0, 1.0, 0.5]},
    ]

    selected = builder.select_target_records(
        records,
        min_public_score=1.0,
        include_non_ties=False,
        limit=None,
        target_priority="input-order",
    )

    assert [record["question_id"] for record in selected] == [
        "all-pass",
        "fragile",
        "three-pass",
    ]


def test_select_target_records_public_fragility_prioritizes_tight_ties() -> None:
    builder = load_prompt_builder_module()
    records = [
        {"question_id": "all-pass", "public_scores": [1.0, 1.0, 1.0, 1.0]},
        {"question_id": "fragile", "public_scores": [1.0, 0.5, 1.0, 0.0]},
        {"question_id": "three-pass", "public_scores": [1.0, 1.0, 1.0, 0.5]},
    ]

    selected = builder.select_target_records(
        records,
        min_public_score=1.0,
        include_non_ties=False,
        limit=2,
        target_priority="public-fragility",
    )

    assert [record["question_id"] for record in selected] == [
        "fragile",
        "three-pass",
    ]


def test_select_target_records_public_ambiguity_prioritizes_many_ties() -> None:
    builder = load_prompt_builder_module()
    records = [
        {"question_id": "fragile", "public_scores": [1.0, 0.5, 1.0, 0.0]},
        {"question_id": "all-pass", "public_scores": [1.0, 1.0, 1.0, 1.0]},
    ]

    selected = builder.select_target_records(
        records,
        min_public_score=1.0,
        include_non_ties=False,
        limit=1,
        target_priority="public-ambiguity",
    )

    assert [record["question_id"] for record in selected] == ["all-pass"]
