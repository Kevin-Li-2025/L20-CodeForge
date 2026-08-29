from __future__ import annotations

from l20_codeforge.training.grpo import _attach_replay_rows


def test_attach_replay_rows_is_seeded_and_deterministic() -> None:
    grpo = [{"task_id": f"task-{index}"} for index in range(5)]
    replay = [
        {
            "messages": [
                {"role": "user", "content": f"prompt-{index}"},
                {"role": "assistant", "content": f"answer-{index}"},
            ]
        }
        for index in range(3)
    ]

    first = _attach_replay_rows(grpo, replay, seed=42)
    second = _attach_replay_rows(grpo, list(reversed(replay)), seed=42)

    assert first == second
    assert all(row["replay_messages"][-1]["role"] == "assistant" for row in first)
    assert len({row["replay_messages"][-1]["content"] for row in first}) == 3
