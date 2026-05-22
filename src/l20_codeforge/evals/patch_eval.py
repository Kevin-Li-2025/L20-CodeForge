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
    visible_test_stdout: str = ""
    visible_test_stderr: str = ""
    hidden_test_stdout: str = ""
    hidden_test_stderr: str = ""


def load_task(path: Path) -> TaskSpec:
    return TaskSpec.model_validate_json(path.read_text(encoding="utf-8"))


def evaluate_patch(
    task: TaskSpec,
    patch: str,
    keep_worktree: bool = False,
    run_hidden: bool = False,
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

    visible_result = None
    if apply_result.returncode == 0 and task.visible_test_command:
        test_env = RepoEnv(worktree, timeout_seconds=timeout_seconds)
        visible_result = test_env.run(task.visible_test_command)
        steps.append(
            AgentStep(
                index=1,
                action=task.visible_test_command,
                observation=(visible_result.stdout + visible_result.stderr)[-20000:],
                exit_code=visible_result.exit_code,
                elapsed_timeout=visible_result.elapsed_timeout,
            )
        )

    hidden_result = None
    if (
        apply_result.returncode == 0
        and run_hidden
        and task.hidden_test_command
        and (visible_result is None or visible_result.exit_code == 0)
    ):
        test_env = RepoEnv(worktree, timeout_seconds=timeout_seconds)
        hidden_result = test_env.run(task.hidden_test_command)
        steps.append(
            AgentStep(
                index=len(steps),
                action=task.hidden_test_command,
                observation=(hidden_result.stdout + hidden_result.stderr)[-20000:],
                exit_code=hidden_result.exit_code,
                elapsed_timeout=hidden_result.elapsed_timeout,
            )
        )

    visible_exit_code = visible_result.exit_code if visible_result else 1
    hidden_exit_code = hidden_result.exit_code if hidden_result else 0
    test_exit_code = 0 if visible_exit_code == 0 and hidden_exit_code == 0 else 1
    reward = score_patch(test_exit_code=test_exit_code, patch=patch)
    if apply_result.returncode != 0:
        status = "invalid"
        reward.notes.append("patch did not apply")
        reward.recompute_total()
    elif visible_exit_code == 0 and hidden_exit_code == 0:
        status = "success"
    elif visible_exit_code == 0:
        status = "partial"
        reward.notes.append("visible tests passed; hidden tests failed")
        reward.recompute_total()
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
        visible_test_stdout=visible_result.stdout if visible_result else "",
        visible_test_stderr=visible_result.stderr if visible_result else "",
        hidden_test_stdout=hidden_result.stdout if hidden_result else "",
        hidden_test_stderr=hidden_result.stderr if hidden_result else "",
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
