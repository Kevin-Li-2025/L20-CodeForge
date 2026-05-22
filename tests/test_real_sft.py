from __future__ import annotations

import json
from pathlib import Path

from l20_codeforge.data.io import write_jsonl
from l20_codeforge.data.real_datasets import RealTaskRecord
from l20_codeforge.data.real_sft import build_real_sft_jsonl, real_task_to_sft_record


def test_real_task_to_sft_record_uses_gold_patch_without_leaking_test_patch() -> None:
    record = RealTaskRecord(
        dataset="swe-bench-lite",
        split="test",
        instance_id="org__repo-1",
        repo="org/repo",
        base_commit="abc",
        problem_statement="Fix a bug.",
        patch="--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-bad\n+good\n",
        test_patch="--- a/tests.py\n+++ b/tests.py\n",
        fail_to_pass=["tests/test_a.py::test_bug"],
        pass_to_pass=["tests/test_a.py::test_existing"],
    )

    sft = real_task_to_sft_record(record)
    payload = sft.model_dump_json()

    assert sft.messages[-1]["content"].startswith("--- a/a.py")
    assert "tests/test_a.py::test_bug" in sft.messages[1]["content"]
    assert "tests.py" not in payload


def test_build_real_sft_jsonl_filters_empty_patches(tmp_path: Path) -> None:
    real_tasks = tmp_path / "real.jsonl"
    write_jsonl(
        real_tasks,
        [
            RealTaskRecord(
                dataset="swe-bench-lite",
                split="test",
                instance_id="ok",
                repo="org/repo",
                problem_statement="Fix a bug.",
                patch="--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-bad\n+good\n",
            ),
            RealTaskRecord(
                dataset="swe-bench-lite",
                split="test",
                instance_id="skip",
                repo="org/repo",
                problem_statement="No patch.",
                patch="",
            ),
        ],
    )
    output = tmp_path / "sft.jsonl"

    count = build_real_sft_jsonl(real_tasks, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert count == 1
    assert rows[0]["instance_id"] == "ok"

