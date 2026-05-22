from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from l20_codeforge.data.io import read_jsonl, write_jsonl
from l20_codeforge.data.schema import Trajectory


SYSTEM_PROMPT = (
    "You are L20 CodeForge, a careful coding agent. Read the issue and produce a "
    "minimal unified diff patch. Prefer small, testable changes and do not modify tests "
    "unless the task explicitly asks for test changes."
)


class SFTRecord(BaseModel):
    messages: list[dict[str, str]]
    task_id: str
    reward_total: float
    status: str
    metadata: dict[str, object] = Field(default_factory=dict)


def trajectory_to_sft_record(trajectory: Trajectory) -> SFTRecord:
    visible_tests = trajectory.task.visible_test_command or "not specified"
    user_prompt = "\n".join(
        [
            f"Task ID: {trajectory.task.task_id}",
            "",
            "Issue:",
            trajectory.task.issue,
            "",
            f"Visible test command: {visible_tests}",
            "",
            "Return only a unified diff patch.",
        ]
    )
    return SFTRecord(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": trajectory.patch.strip() + "\n"},
        ],
        task_id=trajectory.task.task_id,
        reward_total=trajectory.reward.total,
        status=trajectory.status,
        metadata={
            "tests_passed": trajectory.reward.tests_passed,
            "patch_quality": trajectory.reward.patch_quality,
            "anti_hack": trajectory.reward.anti_hack,
            "self_verification": trajectory.reward.self_verification,
        },
    )


def build_sft_jsonl(
    trajectories_path: Path,
    output_path: Path,
    min_reward: float = 1.0,
    include_partial: bool = False,
) -> int:
    allowed_status = {"success"}
    if include_partial:
        allowed_status.add("partial")

    records = []
    for trajectory in read_jsonl(trajectories_path, Trajectory):
        if trajectory.status not in allowed_status:
            continue
        if trajectory.reward.total < min_reward:
            continue
        if not trajectory.patch.strip():
            continue
        records.append(trajectory_to_sft_record(trajectory))

    return write_jsonl(output_path, records)

