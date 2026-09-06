from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_selection_module():
    path = Path(__file__).parents[1] / "scripts" / "select_retention_aware_rlvr_v2.py"
    spec = importlib.util.spec_from_file_location("select_retention_aware_rlvr_v2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_outcomes(path: Path, passing: set[int], *, prefix: str, tasks: int) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": f"{prefix}-{index}",
                    "sample_index": 0,
                    "all_passed": index in passing,
                }
            )
            + "\n"
            for index in range(tasks)
        ),
        encoding="utf-8",
    )


def test_selects_best_eligible_seed_with_retention_gate(tmp_path: Path) -> None:
    module = load_selection_module()
    base_target = tmp_path / "base-target.jsonl"
    base_retention = tmp_path / "base-retention.jsonl"
    write_outcomes(base_target, {0, 1, 2}, prefix="target", tasks=10)
    write_outcomes(base_retention, {0, 1}, prefix="retention", tasks=5)
    candidates = {}
    for name, target_passes, retention_passes in (
        ("seed42", {0, 1, 2, 3}, {0}),
        ("seed43", {0, 1, 2, 3}, {0, 1}),
        ("seed44", {0, 1, 2, 3, 4}, {0, 1}),
    ):
        target = tmp_path / f"{name}-target.jsonl"
        retention = tmp_path / f"{name}-retention.jsonl"
        write_outcomes(target, target_passes, prefix="target", tasks=10)
        write_outcomes(retention, retention_passes, prefix="retention", tasks=5)
        candidates[name] = (target, retention)

    report = module.build_selection_report(
        base_target,
        base_retention,
        candidates,
        expected_target_tasks=10,
        expected_retention_tasks=5,
        required_absolute_delta=0.1,
    )

    assert report["required_target_passes"] == 4
    assert report["selected"]["name"] == "seed44"
    assert report["candidates"][0]["target_gate_passed"] is True
    assert report["candidates"][0]["retention_gate_passed"] is False
    assert "EvalPlus" in report["claim_boundary"]
    assert "+10.0 target points" in report["protocol"]


def test_rejects_retention_task_mismatch(tmp_path: Path) -> None:
    module = load_selection_module()
    base_target = tmp_path / "base-target.jsonl"
    base_retention = tmp_path / "base-retention.jsonl"
    candidate_target = tmp_path / "candidate-target.jsonl"
    candidate_retention = tmp_path / "candidate-retention.jsonl"
    write_outcomes(base_target, {0}, prefix="target", tasks=2)
    write_outcomes(base_retention, {0}, prefix="retention", tasks=2)
    write_outcomes(candidate_target, {0}, prefix="target", tasks=2)
    write_outcomes(candidate_retention, {0}, prefix="wrong", tasks=2)

    with pytest.raises(ValueError, match="retention task set differs"):
        module.build_selection_report(
            base_target,
            base_retention,
            {"seed42": (candidate_target, candidate_retention)},
            expected_target_tasks=2,
            expected_retention_tasks=2,
            required_absolute_delta=0.0,
        )


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_rejects_non_boolean_outcomes(tmp_path: Path, value: object) -> None:
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps({"task_id": "a", "all_passed": value}) + "\n")
    with pytest.raises(ValueError, match="JSON boolean"):
        load_selection_module().load_greedy_outcomes(path)


@pytest.mark.parametrize("index", [1, -1, "0", False, 0.0])
def test_rejects_non_greedy_sample_indices(tmp_path: Path, index: object) -> None:
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps({"task_id": "a", "sample_index": index, "all_passed": True}))
    with pytest.raises(ValueError, match="sample_index=0"):
        load_selection_module().load_greedy_outcomes(path)


def test_rejects_duplicate_greedy_tasks(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.jsonl"
    path.write_text('{"task_id":"a","all_passed":true}\n' * 2)
    with pytest.raises(ValueError, match="duplicate greedy task"):
        load_selection_module().load_greedy_outcomes(path)


@pytest.mark.parametrize("task_id", [None, 3, "", "  "])
def test_rejects_invalid_task_ids(tmp_path: Path, task_id: object) -> None:
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps({"task_id": task_id, "all_passed": False}))
    with pytest.raises(ValueError, match="nonempty string"):
        load_selection_module().load_greedy_outcomes(path)


@pytest.mark.parametrize("delta", [float("nan"), float("inf"), -0.1, 1.1])
def test_rejects_invalid_delta_before_reading_files(tmp_path: Path, delta: float) -> None:
    path = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="required_absolute_delta"):
        load_selection_module().build_selection_report(
            path, path, {"seed42": (path, path)}, required_absolute_delta=delta
        )


def test_accepts_legacy_single_sample_without_index(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.jsonl"
    path.write_text('{"task_id":"a","all_passed":false}\n')
    assert load_selection_module().load_greedy_outcomes(path) == {"a": False}
