from __future__ import annotations

import json
from pathlib import Path

from l20_codeforge.evals.code_rlvr import (
    build_verified_sft_from_rollouts,
    merge_code_rollouts,
    select_mixed_reward_tasks,
    summarize_code_rollouts,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _rollout(task_id: str, sample: int, passed: bool) -> dict[str, object]:
    return {
        "task_id": task_id,
        "sample_index": sample,
        "prompt": f"Solve {task_id}",
        "completion": "print(int(input()) + 1)",
        "code": "print(int(input()) + 1)",
        "all_passed": passed,
        "pass_fraction": 1.0 if passed else 0.5,
        "prompt_sha256": "p",
        "tests_sha256": "t",
    }


def test_summarize_rollouts_separates_greedy_and_pass_at_n() -> None:
    rows = [_rollout("a", 0, False), _rollout("a", 1, True), _rollout("b", 0, True)]
    summary = summarize_code_rollouts(rows, expected_tasks=2)
    assert summary["greedy_accuracy"] == 0.5
    assert summary["pass_at_n"] == 1.0
    assert summary["mixed_reward_tasks"] == 1


def test_build_verified_sft_and_select_mixed_tasks(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(
        rollouts,
        [_rollout("a", 0, False), _rollout("a", 1, True), _rollout("b", 0, True)],
    )
    _write_jsonl(
        tasks,
        [
            {
                "task_id": "a",
                "prompt": "Solve a",
                "tests": [{"stdin": "1", "expected_stdout": "2"}],
            },
            {
                "task_id": "b",
                "prompt": "Solve b",
                "tests": [{"stdin": "1", "expected_stdout": "2"}],
            },
        ],
    )
    sft_report = build_verified_sft_from_rollouts(rollouts, tmp_path / "sft.jsonl")
    mixed_report = select_mixed_reward_tasks(tasks, rollouts, tmp_path / "mixed.jsonl")
    assert sft_report["records"] == 2
    assert mixed_report["selected_tasks"] == 1
    assert json.loads((tmp_path / "mixed.jsonl").read_text().splitlines()[0])["task_id"] == "a"


def test_merge_code_rollouts_rejects_no_valid_rows(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [_rollout("a", 0, True)])
    _write_jsonl(second, [_rollout("b", 0, False)])
    report = merge_code_rollouts([first, second], tmp_path / "merged.jsonl", expected_tasks=2)
    assert report["summary"]["tasks"] == 2
    assert report["summary"]["greedy_accuracy"] == 0.5
