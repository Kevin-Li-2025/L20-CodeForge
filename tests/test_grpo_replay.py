from __future__ import annotations

import pytest

from l20_codeforge.training.grpo import _attach_replay_rows, _chat_template_token_ids


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


class _TemplateTokenizer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.result


def test_chat_template_token_ids_forces_flat_non_dict_output() -> None:
    tokenizer = _TemplateTokenizer([10, 11, 12])
    messages = [{"role": "user", "content": "hello"}]

    assert _chat_template_token_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
    ) == [10, 11, 12]
    assert tokenizer.calls == [
        (
            messages,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": False,
            },
        )
    ]


def test_chat_template_token_ids_rejects_batch_encoding_shape() -> None:
    tokenizer = _TemplateTokenizer({"input_ids": [10, 11]})

    with pytest.raises(TypeError, match="flat token-id list"):
        _chat_template_token_ids(
            tokenizer,
            [{"role": "user", "content": "hello"}],
            add_generation_prompt=False,
        )
