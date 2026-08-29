from __future__ import annotations

import json
from pathlib import Path

from l20_codeforge.training.sft import _load_sft_rows


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "\nassistant: "
        return rendered


def test_load_sft_rows_applies_chat_template(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "patch"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _load_sft_rows(path, DummyTokenizer(), limit=None)  # type: ignore[arg-type]

    assert rows == [{"text": "system: sys\nuser: issue\nassistant: patch"}]


def test_load_sft_rows_can_mask_the_prompt_for_verified_completion_training(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sft.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "patch"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _load_sft_rows(  # type: ignore[arg-type]
        path,
        DummyTokenizer(),
        limit=None,
        completion_only_loss=True,
    )

    assert rows == [
        {
            "prompt": "system: sys\nuser: issue\nassistant: ",
            "completion": "patch",
        }
    ]
