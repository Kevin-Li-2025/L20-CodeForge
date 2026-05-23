from __future__ import annotations

import importlib.util
from pathlib import Path


def load_compare_module():
    script = Path(__file__).parents[1] / "scripts" / "compare_lcb_selection_runs.py"
    spec = importlib.util.spec_from_file_location("compare_lcb_selection_runs", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_selection_comparison_marks_improvement() -> None:
    compare = load_compare_module()

    summary = compare.summarize_comparison(
        baseline_eval={
            "a": {"question_title": "A", "passed": False},
            "b": {"question_title": "B", "passed": True},
        },
        candidate_eval={
            "a": {"question_title": "A", "passed": True},
            "b": {"question_title": "B", "passed": True},
        },
        baseline_selection={
            "a": {"selected_index": 0, "selected_public_score": 1.0},
            "b": {"selected_index": 0, "selected_public_score": 1.0},
        },
        candidate_selection={
            "a": {
                "selected_index": 1,
                "selected_public_score": 1.0,
                "override_from_public": True,
                "behavior_success_rate": 1.0,
            },
            "b": {"selected_index": 0, "selected_public_score": 1.0},
        },
    )

    assert summary["status"] == "improved"
    assert summary["net_gain"] == 1
    assert summary["selection_changed_count"] == 1
    assert summary["changed_selection_improvement_count"] == 1


def test_selection_comparison_marks_unchanged_flips_as_unstable() -> None:
    compare = load_compare_module()

    summary = compare.summarize_comparison(
        baseline_eval={"a": {"question_title": "A", "passed": True}},
        candidate_eval={"a": {"question_title": "A", "passed": False}},
        baseline_selection={"a": {"selected_index": 0}},
        candidate_selection={"a": {"selected_index": 0}},
    )

    assert summary["status"] == "unstable_replay"
    assert summary["unchanged_outcome_flip_count"] == 1
    assert summary["unchanged_outcome_flips"] == ["a"]
