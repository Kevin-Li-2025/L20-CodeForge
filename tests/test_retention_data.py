from __future__ import annotations

import json
from pathlib import Path

from l20_codeforge.data.retention import (
    build_lcb_verified_trajectory_sft,
    compose_retention_sft_mixture,
    materialize_mbpp_replay,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mbpp_row(task_id: int) -> dict[str, object]:
    return {
        "task_id": task_id,
        "text": f"Write a function f_{task_id} that returns its input.",
        "code": f"def f_{task_id}(x):\n    return x",
        "test_list": [f"assert f_{task_id}(3) == 3"],
        "test_setup_code": "",
    }


def test_mbpp_replay_admits_only_official_train_and_validation(tmp_path: Path) -> None:
    rows = (
        [_mbpp_row(task_id) for task_id in range(11, 511)]
        + [_mbpp_row(task_id) for task_id in range(511, 601)]
        + [_mbpp_row(task_id) for task_id in range(601, 975)]
    )
    report = materialize_mbpp_replay(tmp_path, rows=rows)

    train = [json.loads(line) for line in (tmp_path / "train-replay.jsonl").read_text().splitlines()]
    validation = [
        json.loads(line)
        for line in (tmp_path / "validation-retention.jsonl").read_text().splitlines()
    ]
    assert report["train_records"] == len(train) == 374
    assert report["validation_tasks"] == len(validation) == 90
    assert report["excluded_test_ids_seen"] == 500
    assert {row["metadata"]["source_task_id"] for row in train} == set(range(601, 975))
    assert {int(row["task_id"].split("/")[-1]) for row in validation} == set(
        range(511, 601)
    )
    assert not any("mbpp-test" in json.dumps(row) for row in train + validation)


def test_lcb_trajectory_builder_requires_date_and_full_harness_pass(tmp_path: Path) -> None:
    source = tmp_path / "eval_all.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question_id": "old-pass",
                    "question_content": "Print one.",
                    "contest_date": "2024-01-02T00:00:00",
                    "code_list": ["print(1)", "print(2)"],
                    "graded_list": [True, False],
                    "difficulty": "easy",
                    "platform": "codeforces",
                },
                {
                    "question_id": "old-fail",
                    "question_content": "Print two.",
                    "contest_date": "2024-02-02T00:00:00",
                    "code_list": ["print(1)"],
                    "graded_list": [False],
                },
                {
                    "question_id": "future-pass",
                    "question_content": "Print three.",
                    "contest_date": "2025-02-02T00:00:00",
                    "code_list": ["print(3)"],
                    "graded_list": [True],
                },
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "lcb.jsonl"
    report = build_lcb_verified_trajectory_sft(
        source,
        output,
        max_contest_date="2024-08-31T23:59:59",
        max_records=1,
    )

    row = json.loads(output.read_text())
    assert report["eligible_verified_records"] == 1
    assert row["metadata"]["question_id"] == "old-pass"
    assert row["metadata"]["verification"] == "full_livecodebench_harness_passed"
    assert row["messages"][-1]["content"] == "print(1)\n"


def test_retention_sft_mixture_is_deterministic_and_balanced(tmp_path: Path) -> None:
    paths = []
    for source_name in ("target", "lcb", "mbpp"):
        path = tmp_path / f"{source_name}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "messages": [
                        {"role": "user", "content": f"{source_name}-{index}"},
                        {"role": "assistant", "content": "pass"},
                    ],
                    "metadata": {"task_id": f"{source_name}/{index}"},
                }
                for index in range(4)
            ],
        )
        paths.append(path)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_report = compose_retention_sft_mixture(
        *paths,
        first,
        target_records=2,
        lcb_records=2,
        mbpp_records=2,
        seed=7,
    )
    second_report = compose_retention_sft_mixture(
        *paths,
        second,
        target_records=2,
        lcb_records=2,
        mbpp_records=2,
        seed=7,
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_report["output_sha256"] == second_report["output_sha256"]
    assert first_report["source_counts"] == {
        "rstar_target": 2,
        "lcb_trajectory_replay": 2,
        "mbpp_function_replay": 2,
    }
