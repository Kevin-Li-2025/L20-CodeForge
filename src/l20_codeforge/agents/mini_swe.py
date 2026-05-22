from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from l20_codeforge.data.io import write_jsonl
from l20_codeforge.data.schema import AgentStep, RewardBreakdown, TaskSpec, Trajectory
from l20_codeforge.evals.patch_eval import evaluate_patch, load_task


PATCH_START_RE = re.compile(r"^(diff --git |--- a/)", re.MULTILINE)
FENCE_RE = re.compile(r"```(?:diff|patch)?\n(.*?)```", re.DOTALL)


class MiniTaskRecord(BaseModel):
    task_id: str
    task_file: str
    repo: str
    prompt: str
    suggested_output: str
    suggested_command: str


class MiniConversionResult(BaseModel):
    trajectory: Trajectory
    patch_found: bool
    mini_exit_status: str = ""
    mini_submission: str = ""


def build_mini_task_prompt(task: TaskSpec) -> str:
    visible_tests = task.visible_test_command or "not specified"
    return "\n".join(
        [
            task.issue,
            "",
            "You are working in the repository root configured as the agent cwd.",
            "Modify only source files needed to fix the issue. Do not edit tests.",
            f"Run the visible tests with: `{visible_tests}`",
            "Hidden tests will be run after submission.",
            "",
            "When you are done, print the final patch with:",
            "`python3 .l20_codeforge/make_patch.py`",
            "",
            "Then submit by running exactly:",
            "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`",
        ]
    )


def export_mini_task_records(
    task_files: list[Path],
    output_path: Path,
    output_dir: Path = Path("artifacts/mini_swe"),
) -> int:
    records = []
    for task_file in task_files:
        task = load_task(task_file)
        mini_output = output_dir / f"{task.task_id}.traj.json"
        suggested_command = (
            "mini "
            f"--task {shlex.quote(build_mini_task_prompt(task))} "
            "--yolo "
            f"--output {shlex.quote(str(mini_output))} "
            f"--config {shlex.quote(f'environment.cwd={task.repo}')} "
            "--config agent.cost_limit=0"
        )
        records.append(
            MiniTaskRecord(
                task_id=task.task_id,
                task_file=str(task_file),
                repo=task.repo,
                prompt=build_mini_task_prompt(task),
                suggested_output=str(mini_output),
                suggested_command=suggested_command,
            )
        )
    return write_jsonl(output_path, records)


def convert_mini_trajectory_file(
    task_file: Path,
    mini_trajectory_file: Path,
    run_hidden: bool = True,
    timeout_seconds: int = 120,
) -> MiniConversionResult:
    task = load_task(task_file)
    mini_data = json.loads(mini_trajectory_file.read_text(encoding="utf-8"))
    return convert_mini_trajectory_data(
        task=task,
        mini_data=mini_data,
        run_hidden=run_hidden,
        timeout_seconds=timeout_seconds,
    )


def convert_mini_trajectory_data(
    task: TaskSpec,
    mini_data: dict[str, Any],
    run_hidden: bool = True,
    timeout_seconds: int = 120,
) -> MiniConversionResult:
    mini_steps = extract_mini_steps(mini_data)
    info = mini_data.get("info", {})
    mini_exit_status = str(info.get("exit_status", ""))
    mini_submission = str(info.get("submission", ""))
    patch = extract_patch(mini_submission) or extract_patch_from_messages(mini_data.get("messages", []))

    if not patch:
        trajectory = Trajectory(
            task=task,
            status="failed",
            steps=mini_steps,
            patch="",
            reward=RewardBreakdown(notes=["mini trajectory did not contain a unified diff patch"]),
        )
        return MiniConversionResult(
            trajectory=trajectory,
            patch_found=False,
            mini_exit_status=mini_exit_status,
            mini_submission=mini_submission,
        )

    eval_result = evaluate_patch(
        task=task,
        patch=patch,
        run_hidden=run_hidden,
        timeout_seconds=timeout_seconds,
    )
    eval_steps = []
    offset = len(mini_steps)
    for index, step in enumerate(eval_result.trajectory.steps, start=offset):
        step.index = index
        eval_steps.append(step)
    eval_result.trajectory.steps = mini_steps + eval_steps
    return MiniConversionResult(
        trajectory=eval_result.trajectory,
        patch_found=True,
        mini_exit_status=mini_exit_status,
        mini_submission=mini_submission,
    )


def extract_mini_steps(mini_data: dict[str, Any]) -> list[AgentStep]:
    steps: list[AgentStep] = []
    messages = mini_data.get("messages", [])
    pending_actions: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        extra = message.get("extra", {})
        if isinstance(extra, dict) and extra.get("actions"):
            for action in extra.get("actions", []):
                if isinstance(action, dict):
                    pending_actions.append(str(action.get("command", "")))
        raw_output = ""
        returncode = None
        if isinstance(extra, dict):
            raw_output = str(extra.get("raw_output", ""))
            returncode = extra.get("returncode")
        if raw_output or returncode is not None:
            action = pending_actions.pop(0) if pending_actions else "observe"
            steps.append(
                AgentStep(
                    index=len(steps),
                    action=action,
                    observation=raw_output[-20000:],
                    exit_code=returncode if isinstance(returncode, int) else None,
                )
            )
    for action in pending_actions:
        steps.append(
            AgentStep(
                index=len(steps),
                action=action,
                observation="",
                exit_code=None,
            )
        )
    return steps


def extract_patch_from_messages(messages: list[Any]) -> str:
    texts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        extra = message.get("extra", {})
        if isinstance(extra, dict):
            for key in ("raw_output", "model_response", "submission"):
                value = extra.get(key)
                if isinstance(value, str):
                    texts.append(value)
    for text in reversed(texts):
        patch = extract_patch(text)
        if patch:
            return patch
    return ""


def extract_patch(text: str) -> str:
    if not text:
        return ""
    for match in FENCE_RE.finditer(text):
        patch = _extract_patch_tail(match.group(1))
        if patch:
            return patch
    return _extract_patch_tail(text)


def _extract_patch_tail(text: str) -> str:
    match = PATCH_START_RE.search(text)
    if not match:
        return ""
    patch = text[match.start() :].strip()
    return patch + "\n"
