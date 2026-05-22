from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from l20_codeforge.data.io import read_jsonl, write_json
from l20_codeforge.data.schema import Trajectory


class TrajectoryReport(BaseModel):
    trajectories: int
    status_counts: dict[str, int]
    tag_counts: dict[str, int]
    reward_min: float | None = None
    reward_max: float | None = None
    reward_mean: float | None = None
    successful_task_ids: list[str] = Field(default_factory=list)


def build_trajectory_report(trajectories_path: Path) -> TrajectoryReport:
    status_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    rewards: list[float] = []
    successful_task_ids: list[str] = []

    for trajectory in read_jsonl(trajectories_path, Trajectory):
        status_counts[trajectory.status] += 1
        rewards.append(trajectory.reward.total)
        if trajectory.status == "success":
            successful_task_ids.append(trajectory.task.task_id)
        tags = trajectory.task.metadata.get("tags", "")
        if tags:
            for tag in tags.split(","):
                if tag:
                    tag_counts[tag] += 1

    return TrajectoryReport(
        trajectories=sum(status_counts.values()),
        status_counts=dict(status_counts),
        tag_counts=dict(tag_counts),
        reward_min=min(rewards) if rewards else None,
        reward_max=max(rewards) if rewards else None,
        reward_mean=(sum(rewards) / len(rewards)) if rewards else None,
        successful_task_ids=successful_task_ids,
    )


def write_trajectory_report(trajectories_path: Path, output_path: Path) -> TrajectoryReport:
    report = build_trajectory_report(trajectories_path)
    write_json(output_path, report)
    return report

