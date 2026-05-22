from __future__ import annotations

import json

import pytest

from l20_codeforge.data.real_datasets import (
    RealDatasetSpec,
    get_real_dataset_spec,
    list_real_dataset_specs,
    normalize_real_row,
)


def test_real_dataset_registry_prioritizes_swe_bench_lite() -> None:
    specs = list_real_dataset_specs()

    assert specs[0].key == "swe-bench-lite"
    assert get_real_dataset_spec("swe-bench-verified").hf_id == "SWE-bench/SWE-bench_Verified"


def test_unknown_real_dataset_error_is_actionable() -> None:
    with pytest.raises(ValueError, match="known datasets"):
        get_real_dataset_spec("not-a-dataset")


def test_normalize_swe_bench_like_row() -> None:
    spec = RealDatasetSpec(
        key="example",
        hf_id="owner/dataset",
        default_split="test",
        description="example",
        source_url="https://example.com",
        task_type="real_github_issue_pr",
        language="python",
        executable=True,
        license_note="example",
        priority=99,
    )
    row = {
        "instance_id": "org__repo-1",
        "repo": "org/repo",
        "base_commit": "abc123",
        "problem_statement": "Fix the bug",
        "patch": "--- a/a.py\n+++ b/a.py\n",
        "test_patch": "--- a/tests.py\n+++ b/tests.py\n",
        "FAIL_TO_PASS": ["tests/test_a.py::test_bug"],
        "PASS_TO_PASS": ["tests/test_a.py::test_existing"],
        "issue_url": "https://github.com/org/repo/issues/1",
        "pr_url": "https://github.com/org/repo/pull/2",
        "difficulty": "medium",
    }

    record = normalize_real_row(spec, "test", row)

    assert record.instance_id == "org__repo-1"
    assert record.repo == "org/repo"
    assert record.fail_to_pass == ["tests/test_a.py::test_bug"]
    assert record.metadata["difficulty"] == "medium"
    assert json.loads(record.model_dump_json())["dataset"] == "example"
