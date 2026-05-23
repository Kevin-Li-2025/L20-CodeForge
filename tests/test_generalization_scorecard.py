from __future__ import annotations

import importlib.util
from pathlib import Path


def load_scorecard_module():
    script = Path(__file__).parents[1] / "scripts" / "build_generalization_scorecard.py"
    spec = importlib.util.spec_from_file_location("build_generalization_scorecard", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lcb_generalization_builds_slice_gains() -> None:
    scorecard = load_scorecard_module()

    result = scorecard.lcb_generalization(
        {
            "score": {"passed": 12, "total": 20, "pass_at_1": 0.6},
            "baseline_greedy": {"passed": 10, "total": 20, "pass_at_1": 0.5},
            "breakdown": {
                "difficulty": {
                    "easy": {"passed": 8, "total": 10, "score": 0.8},
                    "hard": {"passed": 4, "total": 10, "score": 0.4},
                },
                "platform": {
                    "leetcode": {"passed": 12, "total": 20, "score": 0.6},
                },
            },
            "baseline_breakdown": {
                "difficulty": {
                    "easy": {"passed": 7, "total": 10, "score": 0.7},
                    "hard": {"passed": 3, "total": 10, "score": 0.3},
                },
                "platform": {
                    "leetcode": {"passed": 10, "total": 20, "score": 0.5},
                },
            },
        }
    )

    assert result["overall"]["delta"] == 0.1
    assert result["difficulty"][0]["slice"] == "easy"
    assert result["difficulty"][0]["delta"] == 0.1
    assert result["platform"][0]["delta"] == 0.1


def test_evalplus_generalization_compares_clean_system_to_greedy() -> None:
    scorecard = load_scorecard_module()

    result = scorecard.evalplus_generalization(
        [
            {
                "name": "humaneval_greedy",
                "dataset": "humaneval",
                "protocol": "greedy_model",
                "base_pass@1": 0.89,
                "plus_pass@1": 0.848,
            },
            {
                "name": "humaneval_clean_system_best",
                "dataset": "humaneval",
                "protocol": "clean_public_signal_system",
                "base_pass@1": 0.982,
                "plus_pass@1": 0.927,
            },
        ]
    )

    assert result["comparisons"][0]["dataset"] == "humaneval"
    assert result["comparisons"][0]["plus_pass_at_1"]["delta"] == 0.079


def test_generalization_gate_fails_on_evalplus_regression() -> None:
    scorecard = load_scorecard_module()

    gate = scorecard.evaluate_gate(
        lcb={
            "overall": {"delta": 0.1},
            "difficulty": [{"slice": "easy", "delta": 0.0}],
            "platform": [{"slice": "leetcode", "delta": 0.0}],
        },
        evalplus={
            "comparisons": [
                {
                    "dataset": "humaneval",
                    "plus_pass_at_1": {"delta": -0.05},
                }
            ]
        },
        min_lcb_delta=0.0,
        min_slice_delta=-0.005,
        max_evalplus_plus_regression=0.01,
    )

    assert gate["status"] == "FAIL"
    assert gate["checks"][-1]["passed"] is False
