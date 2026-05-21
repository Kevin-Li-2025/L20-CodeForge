from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class EvalCard(BaseModel):
    name: str
    status: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    source_commit: str | None = None

