from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from l20_codeforge.data.io import write_jsonl


def build_mbpp_sft_jsonl(
    output_path: Path,
    split: str = "train",
    limit: int | None = None,
    exclude_evalplus_mbpp: bool = True,
) -> int:
    rows = load_mbpp_rows(split=split)
    excluded_ids = load_evalplus_mbpp_ids() if exclude_evalplus_mbpp else set()
    records = []
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if task_id and task_id in excluded_ids:
            continue
        records.append(
            {
                "dataset": "mbpp",
                "split": split,
                "task_id": task_id,
                "messages": [
                    {
                        "role": "user",
                        "content": build_mbpp_prompt(row),
                    },
                    {
                        "role": "assistant",
                        "content": str(row.get("code", "")).strip() + "\n",
                    },
                ],
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return write_jsonl(output_path, records)


def load_mbpp_rows(split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    try:
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    except Exception:
        dataset = load_dataset("mbpp", "sanitized", split=split)
    return [dict(row) for row in dataset]


def load_evalplus_mbpp_ids() -> set[str]:
    try:
        from evalplus.data import get_mbpp_plus
    except Exception:
        return set()
    try:
        tasks = get_mbpp_plus()
    except Exception:
        return set()
    return {task_id.split("/")[1] for task_id in tasks}


def build_mbpp_prompt(row: dict[str, Any]) -> str:
    tests = row.get("test_list") or []
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError:
            tests = [tests]
    test_block = "\n".join(str(test) for test in tests[:4])
    return (
        "Solve the following Python programming problem. Return only valid Python code.\n\n"
        f"Problem:\n{row.get('text', '')}\n\n"
        f"Reference tests:\n{test_block}\n"
    )
