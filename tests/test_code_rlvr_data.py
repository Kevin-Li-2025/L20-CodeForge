from __future__ import annotations

import json
from pathlib import Path

from l20_codeforge.data.code_rlvr import (
    jaccard_similarity,
    materialize_rstar_code_rlvr,
    prompt_shingles,
    rstar_row_to_code_task,
)


def _raw_row(index: int, *, tests: int = 10) -> dict[str, str]:
    return {
        "question_id": f"synthetic_{index}",
        "question": (
            f"Problem {index}: Given an integer n, compute the value n plus {index}. "
            "Read one integer and print one integer."
        ),
        "inputs": json.dumps([f"{value}\n" for value in range(tests)]),
        "outputs": json.dumps([f"{value + index}\n" for value in range(tests)]),
    }


def test_rstar_row_to_code_task_is_deterministic_and_bounded() -> None:
    row = _raw_row(3, tests=12)
    first, reason = rstar_row_to_code_task(
        row,
        seed=7,
        min_tests=4,
        max_tests=6,
        max_case_input_chars=100,
        max_case_output_chars=100,
        max_prompt_chars=1000,
    )
    second, _ = rstar_row_to_code_task(
        row,
        seed=7,
        min_tests=4,
        max_tests=6,
        max_case_input_chars=100,
        max_case_output_chars=100,
        max_prompt_chars=1000,
    )
    assert reason == "admitted"
    assert first == second
    assert first is not None
    assert len(first["tests"]) == 6
    assert first["source"]["license"] == "CC-BY-4.0"


def test_materialize_rstar_code_rlvr_freezes_disjoint_receipts(tmp_path: Path) -> None:
    rows = [_raw_row(index) for index in range(12)]
    report = materialize_rstar_code_rlvr(
        tmp_path,
        train_tasks=4,
        dev_tasks=3,
        seed=11,
        min_tests=4,
        max_tests=6,
        rows=rows,
    )
    assert report["train_tasks"] == 4
    assert report["dev_tasks"] == 3
    assert set(report["train_task_ids"]).isdisjoint(report["dev_task_ids"])
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "dev.jsonl").exists()
    assert len(report["train_sha256"]) == 64
    assert len(report["dev_sha256"]) == 64


def test_prompt_shingle_jaccard_detects_near_duplicate() -> None:
    left = prompt_shingles("Read n integers and print their sum in Python.")
    right = prompt_shingles("Read n integers and print their sum in Python please.")
    unrelated = prompt_shingles("Find the shortest path in a weighted graph.")
    assert jaccard_similarity(left, right) > jaccard_similarity(left, unrelated)
