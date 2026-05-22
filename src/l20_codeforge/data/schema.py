from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    task_id: str
    repo: str
    base_commit: str | None = None
    issue: str
    visible_test_command: str | None = None
    hidden_test_command: str | None = None
    allowed_commands: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class AgentStep(BaseModel):
    index: int
    action: str
    observation: str
    exit_code: int | None = None
    elapsed_seconds: float | None = None
    elapsed_timeout: bool = False
    approx_prompt_tokens: int | None = None
    approx_completion_tokens: int | None = None


class RewardBreakdown(BaseModel):
    tests_passed: float = 0.0
    patch_quality: float = 0.0
    anti_hack: float = 0.0
    self_verification: float = 0.0
    total: float = 0.0
    notes: list[str] = Field(default_factory=list)

    def recompute_total(self) -> "RewardBreakdown":
        self.total = (
            self.tests_passed + self.patch_quality + self.anti_hack + self.self_verification
        )
        return self


class Trajectory(BaseModel):
    task: TaskSpec
    status: Literal["success", "partial", "failed", "invalid"]
    steps: list[AgentStep] = Field(default_factory=list)
    patch: str = ""
    reward: RewardBreakdown = Field(default_factory=RewardBreakdown)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(self.model_dump_json() + "\n")
