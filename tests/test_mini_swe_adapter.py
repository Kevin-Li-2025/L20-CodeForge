from __future__ import annotations

import json
import subprocess
from pathlib import Path

from l20_codeforge.agents.mini_swe import (
    build_mini_task_prompt,
    convert_mini_trajectory_file,
    export_mini_task_records,
    extract_mini_steps,
    extract_patch,
)
from l20_codeforge.data.io import read_jsonl
from l20_codeforge.data.preferences import PreferencePair, build_preference_pairs
from l20_codeforge.data.smoke_tasks import write_smoke_tasks
from l20_codeforge.evals.patch_eval import evaluate_patch, load_task


def test_extract_patch_from_fenced_diff() -> None:
    text = """Here is the fix:

```diff
--- a/example.py
+++ b/example.py
@@ -1 +1 @@
-bad
+good
```
"""

    patch = extract_patch(text)

    assert patch.startswith("--- a/example.py")
    assert "+good" in patch


def test_extract_mini_steps_from_tool_observations() -> None:
    mini_data = {
        "messages": [
            {"role": "assistant", "extra": {"actions": [{"command": "ls"}]}},
            {"role": "tool", "extra": {"raw_output": "file.py\n", "returncode": 0}},
        ]
    }

    steps = extract_mini_steps(mini_data)

    assert len(steps) == 1
    assert steps[0].action == "ls"
    assert steps[0].exit_code == 0


def test_convert_mini_trajectory_evaluates_submission_patch(tmp_path: Path) -> None:
    task_file = write_smoke_tasks(tmp_path / "tasks")[0]
    task = load_task(task_file)
    patch = Path(task.metadata["reference_patch"]).read_text(encoding="utf-8")
    mini_path = tmp_path / "mini.traj.json"
    mini_path.write_text(
        json.dumps(
            {
                "info": {"exit_status": "Submitted", "submission": patch},
                "messages": [
                    {"role": "assistant", "extra": {"actions": [{"command": "python3 .l20_codeforge/make_patch.py"}]}},
                    {"role": "tool", "extra": {"raw_output": patch, "returncode": 0}},
                ],
                "trajectory_format": "mini-swe-agent-1.1",
            }
        ),
        encoding="utf-8",
    )

    result = convert_mini_trajectory_file(task_file, mini_path, run_hidden=True)

    assert result.patch_found
    assert result.trajectory.status == "success"
    assert result.trajectory.reward.tests_passed == 1.0
    assert result.trajectory.steps[0].action == "python3 .l20_codeforge/make_patch.py"


def test_export_mini_task_records(tmp_path: Path) -> None:
    task_files = write_smoke_tasks(tmp_path / "tasks")[:2]
    output = tmp_path / "mini_tasks.jsonl"

    count = export_mini_task_records(task_files, output)
    lines = output.read_text(encoding="utf-8").strip().splitlines()

    assert count == 2
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert "python3 .l20_codeforge/make_patch.py" in payload["prompt"]
    assert "mini " in payload["suggested_command"]


def test_patch_helper_emits_reference_patch_after_fix(tmp_path: Path) -> None:
    task_file = write_smoke_tasks(tmp_path / "tasks")[0]
    task = load_task(task_file)
    patch = Path(task.metadata["reference_patch"]).read_text(encoding="utf-8")
    repo = Path(task.repo)

    subprocess.run(["git", "apply", "-"], cwd=repo, input=patch, text=True, check=True)
    generated = subprocess.run(
        ["python3", ".l20_codeforge/make_patch.py"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert generated == patch


def test_build_preference_pairs_from_success_and_failure(tmp_path: Path) -> None:
    task_file = write_smoke_tasks(tmp_path / "tasks")[0]
    task = load_task(task_file)
    success_patch = Path(task.metadata["reference_patch"]).read_text(encoding="utf-8")
    rejected_patch = """--- a/string_utils.py
+++ b/string_utils.py
@@ -3,4 +3,4 @@
 
 def slugify(text: str) -> str:
     \"\"\"Return a URL-friendly slug.\"\"\"
-    return re.sub(r"[^a-zA-Z0-9]+", "-", text.lower())
+    return text.lower()
"""
    trajectories = tmp_path / "trajectories.jsonl"
    evaluate_patch(task, success_patch, run_hidden=True).trajectory.write_jsonl(trajectories)
    evaluate_patch(task, rejected_patch, run_hidden=True).trajectory.write_jsonl(trajectories)

    output = tmp_path / "pairs.jsonl"
    count = build_preference_pairs(trajectories, output)
    pairs = list(read_jsonl(output, PreferencePair))

    assert count == 1
    assert pairs[0].task_id == task.task_id
    assert pairs[0].chosen_reward > pairs[0].rejected_reward


def test_build_mini_prompt_mentions_visible_not_hidden(tmp_path: Path) -> None:
    task = load_task(write_smoke_tasks(tmp_path / "tasks")[0])

    prompt = build_mini_task_prompt(task)

    assert task.visible_test_command in prompt
    assert "Hidden tests will be run after submission" in prompt
    assert task.hidden_test_command not in prompt
