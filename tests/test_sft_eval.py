from __future__ import annotations

import json
from pathlib import Path

from l20_codeforge.evals.sft_eval import (
    is_unified_diff_like,
    load_instance_id_set,
    load_sft_eval_rows,
)


def _write_rows(path: Path, instance_ids: list[str]) -> None:
    rows = [
        {
            "instance_id": instance_id,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "issue"},
                {"role": "assistant", "content": "patch"},
            ],
        }
        for instance_id in instance_ids
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_load_sft_eval_rows_filters_excluded_ids(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    _write_rows(path, ["a", "b", "c"])

    rows = load_sft_eval_rows(path, limit=2, exclude_instance_ids={"a"})

    assert [row["instance_id"] for row in rows] == ["b", "c"]


def test_load_instance_id_set(tmp_path: Path) -> None:
    path = tmp_path / "ids.jsonl"
    _write_rows(path, ["a", "b"])

    assert load_instance_id_set(path) == {"a", "b"}


def test_is_unified_diff_like() -> None:
    assert is_unified_diff_like("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n")
    assert is_unified_diff_like("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n")
    assert not is_unified_diff_like("def f():\n    return 1\n")
