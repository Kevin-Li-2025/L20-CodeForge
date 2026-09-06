from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def load_module():
    script = Path(__file__).parents[1] / "scripts" / "compare_lcb_equal_budget_selectors.py"
    spec = importlib.util.spec_from_file_location("compare_lcb_equal_budget_selectors", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fixtures():
    generations = [
        {"question_id": "a", "code_list": ["a0", "a1"]},
        {"question_id": "b", "code_list": ["b0", "b1"]},
    ]
    evaluations = [
        {"question_id": "a", "code_list": ["a0", "a1"], "graded_list": [False, True]},
        {"question_id": "b", "code_list": ["b0", "b1"], "graded_list": [True, False]},
    ]
    public = [
        {"question_id": "a", "n_candidates": 2, "selected_index": 1},
        {"question_id": "b", "n_candidates": 2, "selected_index": 0},
    ]
    behavior = [
        {"question_id": "a", "n_candidates": 2, "selected_index": 0},
        {"question_id": "b", "n_candidates": 2, "selected_index": 0},
    ]
    return generations, evaluations, public, behavior


def test_equal_budget_comparison_scores_frozen_indices() -> None:
    compare = load_module()
    result = compare.build_comparison(*fixtures(), random_trials=50)

    assert result["budget"] == {
        "tasks": 2,
        "candidates_per_task": 2,
        "total_candidates": 4,
        "candidate_pool_identical_for_rescoring": True,
        "selection_source_hashes_verified": False,
    }
    assert result["fixed_selectors"]["first_candidate"]["passed"] == 1
    assert result["fixed_selectors"]["public_tests"]["passed"] == 2
    assert result["fixed_selectors"]["behavior_consensus"]["passed"] == 1
    assert (
        result["fixed_selectors"]["public_tests"]["selector_success_given_oracle_available"] == 1.0
    )
    assert result["fixed_selectors"]["behavior_consensus"]["missed_available_solutions"] == 1
    assert result["uniform_random_selector"]["analytical_expected_pass_rate"] == 0.5
    assert result["hidden_oracle_ceiling"]["passed"] == 2


def test_mismatched_candidate_order_fails_closed() -> None:
    compare = load_module()
    generations, evaluations, public, behavior = fixtures()
    evaluations[0]["code_list"] = ["a1", "a0"]

    with pytest.raises(ValueError, match="candidate code/order mismatch"):
        compare.build_comparison(generations, evaluations, public, behavior, random_trials=5)


def test_id_or_budget_mismatch_fails_closed() -> None:
    compare = load_module()
    generations, evaluations, public, behavior = fixtures()
    public.pop()
    with pytest.raises(ValueError, match="IDs do not match"):
        compare.build_comparison(generations, evaluations, public, behavior, random_trials=5)


def test_exact_paired_pvalue_handles_ties_and_symmetry() -> None:
    compare = load_module()
    assert compare.exact_paired_binomial_pvalue(0, 0) == 1.0
    assert compare.exact_paired_binomial_pvalue(1, 4) == compare.exact_paired_binomial_pvalue(4, 1)


@pytest.mark.parametrize("grade", ["false", 0, -1, None])
def test_rejects_non_boolean_hidden_grade(grade):
    generations, evaluations, public, behavior = fixtures()
    evaluations[0]["graded_list"][0] = grade
    with pytest.raises(ValueError, match="JSON booleans"):
        load_module().build_comparison(generations, evaluations, public, behavior, random_trials=5)


@pytest.mark.parametrize("index", [-1, 2, 0.5, "0", True])
def test_rejects_invalid_selected_index(index):
    generations, evaluations, public, behavior = fixtures()
    public[0]["selected_index"] = index
    with pytest.raises(ValueError, match="selected index"):
        load_module().build_comparison(generations, evaluations, public, behavior, random_trials=5)


def test_rejects_empty_pool():
    with pytest.raises(ValueError, match="must not be empty"):
        load_module().build_comparison([], [], [], [], random_trials=5)


def test_report_handles_pool_without_any_correct_solution():
    generations, evaluations, public, behavior = fixtures()
    for row in evaluations:
        row["graded_list"] = [False, False]
    module = load_module()
    result = module.build_comparison(generations, evaluations, public, behavior, random_trials=5)
    report = module.render_markdown(result)
    assert "n/a" in report
    assert "21/60" not in report


@pytest.mark.parametrize("report", [{}, {"generations_sha256": "wrong"}])
def test_selection_report_must_bind_exact_generation_pool(report):
    with pytest.raises(ValueError, match="generation hash"):
        load_module().verify_selection_source(report, "a" * 64, "public")


def test_archived_equal_budget_report_rebuilds_exactly(tmp_path):
    root = Path(__file__).parents[1]
    bundle = root / "benchmarks/livecodebench_full_release_v6_2026_05_22"
    expected = json.loads((bundle / "equal_budget_selector_comparison.json").read_text())
    module = load_module()
    args = [sys.executable, str(root / "scripts/compare_lcb_equal_budget_selectors.py")]
    flags = {
        "generations": "--generations",
        "hidden_eval": "--hidden-eval",
        "public_selection": "--public-selection",
        "behavior_selection": "--behavior-selection",
        "generation_report": "--generation-report",
        "public_tests_report": "--public-report",
        "behavior_consensus_report": "--behavior-report",
    }
    for name, flag in flags.items():
        receipt = expected["inputs"][name]
        assert module.sha256_file(root / receipt["path"]) == receipt["sha256"]
        args.extend([flag, receipt["path"]])
    output_json, output_md = tmp_path / "result.json", tmp_path / "result.md"
    args.extend(["--output-json", str(output_json), "--output-markdown", str(output_md)])
    subprocess.run(args, cwd=root, check=True, capture_output=True, text=True)
    assert json.loads(output_json.read_text()) == expected
    assert output_md.read_text() == (bundle / "EQUAL_BUDGET_SELECTOR_COMPARISON.md").read_text()
