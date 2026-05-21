from __future__ import annotations

from typing import Any

from l20_codeforge.rewards.patch_reward import anti_hack_score, patch_quality_score


def patch_quality_reward(completions: list[Any], **_: Any) -> list[float]:
    return [patch_quality_score(_completion_text(item))[0] for item in completions]


def anti_hack_reward(completions: list[Any], **_: Any) -> list[float]:
    return [anti_hack_score(_completion_text(item))[0] for item in completions]


def _completion_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, list) and item and isinstance(item[-1], dict):
        return str(item[-1].get("content", ""))
    if isinstance(item, dict):
        return str(item.get("content", item))
    return str(item)

