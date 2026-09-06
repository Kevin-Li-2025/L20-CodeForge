from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_selection_module():
    path = Path(__file__).parents[1] / "scripts" / "select_code_retention_scale.py"
    spec = importlib.util.spec_from_file_location("select_code_retention_scale", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_outcomes(path: Path, passing: set[int], *, tasks: int = 10) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": f"task-{index}",
                    "sample_index": 0,
                    "all_passed": index in passing,
                }
            )
            + "\n"
            for index in range(tasks)
        ),
        encoding="utf-8",
    )


def test_selects_smallest_scale_meeting_development_gate(tmp_path: Path) -> None:
    module = load_selection_module()
    base = tmp_path / "base.jsonl"
    half = tmp_path / "half.jsonl"
    three_quarters = tmp_path / "three-quarters.jsonl"
    full = tmp_path / "full.jsonl"
    write_outcomes(base, {0, 1, 2})
    write_outcomes(half, {0, 1, 2})
    write_outcomes(three_quarters, {0, 1, 2, 3})
    write_outcomes(full, {0, 1, 2, 3, 4})

    report = module.build_selection_report(
        base,
        {1.0: full, 0.5: half, 0.75: three_quarters},
        expected_tasks=10,
        required_absolute_delta=0.1,
    )

    assert report["required_passed"] == 4
    assert report["selected"]["scale"] == 0.75
    assert [row["scale"] for row in report["candidates"]] == [0.5, 0.75, 1.0]
    assert "EvalPlus" in report["claim_boundary"]


def test_rejects_candidate_task_set_mismatch(tmp_path: Path) -> None:
    module = load_selection_module()
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_outcomes(base, {0}, tasks=2)
    write_outcomes(candidate, {0}, tasks=1)

    with pytest.raises(ValueError, match="task set differs"):
        module.build_selection_report(
            base,
            {1.0: candidate},
            expected_tasks=2,
            required_absolute_delta=0.0,
        )
