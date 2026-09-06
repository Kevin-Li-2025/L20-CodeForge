#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_sort_key(task_id: str) -> tuple[str, int, str]:
    match = re.match(r"^(.*?)/(\d+)$", task_id)
    if match:
        return match.group(1), int(match.group(2)), task_id
    return task_id, -1, task_id


def merge_evalplus_shards(
    inputs: list[Path], output: Path, *, expected_tasks: int
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one input shard is required")
    samples: dict[str, dict[str, Any]] = {}
    input_counts: dict[str, int] = {}
    for path in inputs:
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = str(row["task_id"])
                if task_id in samples:
                    raise ValueError(f"duplicate task_id {task_id} in {path}:{line_number}")
                if not str(row.get("solution", "")).strip():
                    raise ValueError(f"empty solution for {task_id} in {path}:{line_number}")
                samples[task_id] = row
                count += 1
        input_counts[str(path)] = count
    if len(samples) != expected_tasks:
        raise ValueError(f"merged {len(samples)} tasks; expected {expected_tasks}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task_id in sorted(samples, key=task_sort_key):
            handle.write(json.dumps(samples[task_id]) + "\n")
    report = {
        "inputs": [str(path) for path in inputs],
        "input_counts": input_counts,
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "output": str(output),
        "output_sha256": sha256_file(output),
        "tasks": len(samples),
        "expected_tasks": expected_tasks,
        "task_ids": sorted(samples, key=task_sort_key),
    }
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--expected-tasks", type=int, required=True)
    args = parser.parse_args()
    report = merge_evalplus_shards(
        args.inputs,
        args.output,
        expected_tasks=args.expected_tasks,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
