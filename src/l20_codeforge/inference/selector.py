from __future__ import annotations

from pydantic import BaseModel, Field

from l20_codeforge.data.schema import RewardBreakdown


class CandidatePatch(BaseModel):
    candidate_id: str
    patch: str
    reward: RewardBreakdown
    verifier_notes: list[str] = Field(default_factory=list)


def select_best(candidates: list[CandidatePatch]) -> CandidatePatch:
    if not candidates:
        raise ValueError("no candidates provided")
    return max(candidates, key=lambda candidate: candidate.reward.total)

