from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from l20_codeforge.data.io import read_jsonl, write_jsonl
from l20_codeforge.data.sft import SYSTEM_PROMPT
from l20_codeforge.data.schema import Trajectory


class PreferencePair(BaseModel):
    task_id: str
    prompt: list[dict[str, str]]
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    metadata: dict[str, object] = Field(default_factory=dict)


def build_preference_pairs(
    trajectories_path: Path,
    output_path: Path,
    min_reward_gap: float = 0.25,
) -> int:
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in read_jsonl(trajectories_path, Trajectory):
        if trajectory.patch.strip():
            by_task[trajectory.task.task_id].append(trajectory)

    pairs = []
    for task_id, trajectories in by_task.items():
        if len(trajectories) < 2:
            continue
        ranked = sorted(trajectories, key=lambda item: item.reward.total, reverse=True)
        chosen = ranked[0]
        rejected = ranked[-1]
        if chosen.reward.total - rejected.reward.total < min_reward_gap:
            continue
        if chosen.patch.strip() == rejected.patch.strip():
            continue
        pairs.append(_to_preference_pair(task_id, chosen, rejected))

    return write_jsonl(output_path, pairs)


def _to_preference_pair(task_id: str, chosen: Trajectory, rejected: Trajectory) -> PreferencePair:
    visible_tests = chosen.task.visible_test_command or "not specified"
    user_prompt = "\n".join(
        [
            f"Task ID: {task_id}",
            "",
            "Issue:",
            chosen.task.issue,
            "",
            f"Visible test command: {visible_tests}",
            "",
            "Return only a unified diff patch.",
        ]
    )
    return PreferencePair(
        task_id=task_id,
        prompt=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        chosen=chosen.patch.strip() + "\n",
        rejected=rejected.patch.strip() + "\n",
        chosen_reward=chosen.reward.total,
        rejected_reward=rejected.reward.total,
        metadata={
            "chosen_status": chosen.status,
            "rejected_status": rejected.status,
        },
    )

