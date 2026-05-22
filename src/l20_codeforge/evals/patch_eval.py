from __future__ import annotations

import subprocess
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel

from l20_codeforge.data.schema import AgentStep, TaskSpec, Trajectory
from l20_codeforge.envs.repo_env import RepoEnv
from l20_codeforge.rewards.patch_reward import score_patch


class PatchEvalResult(BaseModel):
    trajectory: Trajectory
    worktree: str
    apply_stdout: str = ""
    apply_stderr: str = ""
    test_stdout: str = ""
    test_stderr: str = ""


def load_task(path: Path) -> TaskSpec:
    return TaskSpec.model_validate_json(path.read_text(encoding="utf-8"))


def evaluate_patch(
    task: TaskSpec,
    patch: str,
    keep_worktree: bool = False,
    timeout_seconds: int = 120,
) -> PatchEvalResult:
    source_repo = Path(task.repo).expanduser().resolve()
    env = RepoEnv(source_repo, timeout_seconds=timeout_seconds)
    worktree = env.isolated_copy()

    apply_result = _apply_patch(worktree, patch)
    steps = [
        AgentStep(
            index=0,
            action="apply_patch",
            observation=(apply_result.stdout + apply_result.stderr)[-20000:],
            exit_code=apply_result.returncode,
        )
    ]

    test_result = None
    if apply_result.returncode == 0 and task.visible_test_command:
        test_env = RepoEnv(worktree, timeout_seconds=timeout_seconds)
        test_result = test_env.run(task.visible_test_command)
        steps.append(
            AgentStep(
                index=1,
                action=task.visible_test_command,
                observation=(test_result.stdout + test_result.stderr)[-20000:],
                exit_code=test_result.exit_code,
                elapsed_timeout=test_result.elapsed_timeout,
            )
        )

    test_exit_code = test_result.exit_code if test_result else 1
    reward = score_patch(test_exit_code=test_exit_code, patch=patch)
    if apply_result.returncode != 0:
        status = "invalid"
        reward.notes.append("patch did not apply")
        reward.recompute_total()
    elif test_exit_code == 0:
        status = "success"
    else:
        status = "partial"

    trajectory = Trajectory(
        task=task,
        status=status,
        steps=steps,
        patch=patch,
        reward=reward,
    )
    if not keep_worktree:
        _schedule_cleanup(worktree)

    return PatchEvalResult(
        trajectory=trajectory,
        worktree=str(worktree),
        apply_stdout=apply_result.stdout,
        apply_stderr=apply_result.stderr,
        test_stdout=test_result.stdout if test_result else "",
        test_stderr=test_result.stderr if test_result else "",
    )


def _apply_patch(repo: Path, patch: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=repo,
        input=patch,
        text=True,
        capture_output=True,
        check=False,
    )


def _schedule_cleanup(worktree: Path) -> None:
    # Keep cleanup explicit and local to /tmp to avoid deleting user paths after malformed tasks.
    temp_root = Path(tempfile.gettempdir()).resolve()
    parent = worktree.parent.resolve()
    if parent.parent == temp_root and parent.name.startswith("l20-codeforge-"):
        shutil.rmtree(parent, ignore_errors=True)
