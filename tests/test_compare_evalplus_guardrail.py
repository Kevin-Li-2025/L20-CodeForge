from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_guardrail_module():
    path = Path(__file__).parents[1] / "scripts" / "compare_evalplus_guardrail.py"
    spec = importlib.util.spec_from_file_location("compare_evalplus_guardrail", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_results(path: Path, statuses: list[tuple[str, str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "eval": {
                    f"task-{index}": [
                        {"base_status": base_status, "plus_status": plus_status}
                    ]
                    for index, (base_status, plus_status) in enumerate(statuses)
                }
            }
        ),
        encoding="utf-8",
    )


def test_exact_plus_guardrail_counts_base_and_extra_tests(tmp_path: Path) -> None:
    module = load_guardrail_module()
    base = tmp_path / "base.json"
    final = tmp_path / "final.json"
    write_results(base, [("pass", "pass"), ("pass", "fail"), ("fail", "pass")])
    write_results(final, [("fail", "fail"), ("pass", "pass"), ("pass", "pass")])

    report = module.compare_dataset(base, final, expected_tasks=3)

    assert report["base"]["plus_passed"] == 1
    assert report["final"]["plus_passed"] == 2
    assert report["paired_plus_transitions"] == {
        "gained_tasks": 2,
        "lost_tasks": 1,
        "net_tasks": 1,
    }
    assert report["no_regression"] is True


def test_rejects_non_greedy_result_lists(tmp_path: Path) -> None:
    module = load_guardrail_module()
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "eval": {
                    "task-0": [
                        {"base_status": "pass", "plus_status": "pass"},
                        {"base_status": "pass", "plus_status": "pass"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="greedy n=1"):
        module.load_greedy_evalplus_outcomes(path)
