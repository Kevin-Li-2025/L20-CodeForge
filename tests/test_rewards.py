from __future__ import annotations

from l20_codeforge.rewards.patch_reward import score_patch


def test_score_patch_rewards_passing_tests() -> None:
    patch = "+++ b/a.py\n+def ok():\n+    return True\n"
    reward = score_patch(test_exit_code=0, patch=patch)

    assert reward.tests_passed == 1.0
    assert reward.total > 1.0


def test_score_patch_penalizes_skip_markers() -> None:
    patch = "+import pytest\n+pytest.mark.skip('hide failure')\n"
    reward = score_patch(test_exit_code=0, patch=patch)

    assert reward.anti_hack < 0
    assert reward.notes

