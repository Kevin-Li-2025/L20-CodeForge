from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_merge_module():
    path = Path(__file__).parents[1] / "scripts" / "merge_evalplus_shards.py"
    spec = importlib.util.spec_from_file_location("merge_evalplus_shards", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_samples(path: Path, task_ids: list[str]) -> None:
    path.write_text(
        "".join(
            json.dumps({"task_id": task_id, "solution": f"def f(): return {index}"}) + "\n"
            for index, task_id in enumerate(task_ids)
        ),
        encoding="utf-8",
    )


def test_merges_modulo_shards_in_numeric_task_order(tmp_path: Path) -> None:
    module = load_merge_module()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    write_samples(first, ["Mbpp/2", "Mbpp/10"])
    write_samples(second, ["Mbpp/1", "Mbpp/7"])

    report = module.merge_evalplus_shards([first, second], output, expected_tasks=4)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert [row["task_id"] for row in rows] == ["Mbpp/1", "Mbpp/2", "Mbpp/7", "Mbpp/10"]
    assert report["input_counts"] == {str(first): 2, str(second): 2}
    assert report["tasks"] == 4
    assert output.with_suffix(".report.json").exists()


def test_rejects_duplicate_tasks_across_shards(tmp_path: Path) -> None:
    module = load_merge_module()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_samples(first, ["HumanEval/0"])
    write_samples(second, ["HumanEval/0"])

    with pytest.raises(ValueError, match="duplicate task_id"):
        module.merge_evalplus_shards(
            [first, second], tmp_path / "merged.jsonl", expected_tasks=1
        )
