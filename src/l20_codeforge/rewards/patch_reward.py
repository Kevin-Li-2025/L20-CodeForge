from __future__ import annotations

import re

from l20_codeforge.data.schema import RewardBreakdown


SECRET_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"api[_-]?key\s*=",
        r"password\s*=",
        r"token\s*=",
        r"skip\s+test",
        r"xfail",
        r"pytest\.mark\.skip",
    ]
]


def patch_quality_score(patch: str) -> tuple[float, list[str]]:
    added = sum(1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---"))
    notes: list[str] = []

    if added + removed == 0:
        return 0.0, ["empty patch"]
    if added + removed > 500:
        notes.append("large patch")
        return -0.2, notes
    if removed > added * 4 and removed > 20:
        notes.append("mostly deletion")
        return -0.1, notes

    return 0.15, notes


def anti_hack_score(patch: str) -> tuple[float, list[str]]:
    notes = []
    for pattern in SECRET_PATTERNS:
        if pattern.search(patch):
            notes.append(f"suspicious pattern: {pattern.pattern}")
    return (-0.35 if notes else 0.0), notes


def score_patch(test_exit_code: int, patch: str, self_verification: float = 0.0) -> RewardBreakdown:
    quality, quality_notes = patch_quality_score(patch)
    anti_hack, hack_notes = anti_hack_score(patch)
    reward = RewardBreakdown(
        tests_passed=1.0 if test_exit_code == 0 else 0.0,
        patch_quality=quality,
        anti_hack=anti_hack,
        self_verification=self_verification,
        notes=quality_notes + hack_notes,
    )
    return reward.recompute_total()

