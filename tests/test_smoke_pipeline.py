from __future__ import annotations

from pathlib import Path

from l20_codeforge.data.io import read_jsonl
from l20_codeforge.data.schema import Trajectory
from l20_codeforge.data.sft import build_sft_jsonl
from l20_codeforge.data.smoke_tasks import write_smoke_tasks
from l20_codeforge.evals.patch_eval import evaluate_patch, load_task


def test_smoke_task_reference_patches_pass(tmp_path: Path) -> None:
    task_files = write_smoke_tasks(tmp_path / "tasks")

    assert len(task_files) == 3
    for task_file in task_files:
        task = load_task(task_file)
        patch = Path(task.metadata["reference_patch"]).read_text(encoding="utf-8")
        result = evaluate_patch(task, patch)

        assert result.trajectory.status == "success"
        assert result.trajectory.reward.tests_passed == 1.0
        assert result.trajectory.reward.total >= 1.0


def test_build_sft_jsonl_from_successful_trajectory(tmp_path: Path) -> None:
    task_file = write_smoke_tasks(tmp_path / "tasks")[0]
    task = load_task(task_file)
    patch = Path(task.metadata["reference_patch"]).read_text(encoding="utf-8")
    result = evaluate_patch(task, patch)

    trajectories = tmp_path / "trajectories.jsonl"
    result.trajectory.write_jsonl(trajectories)
    output = tmp_path / "sft.jsonl"

    count = build_sft_jsonl(trajectories, output, min_reward=1.0)
    records = output.read_text(encoding="utf-8").strip().splitlines()

    assert count == 1
    assert len(records) == 1
    assert list(read_jsonl(trajectories, Trajectory))[0].task.task_id == task.task_id

