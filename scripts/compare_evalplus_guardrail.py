#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_greedy_evalplus_outcomes(path: Path) -> dict[str, dict[str, bool]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluation = payload.get("eval")
    if not isinstance(evaluation, dict):
        raise TypeError(f"missing eval mapping in {path}")
    outcomes: dict[str, dict[str, bool]] = {}
    for task_id, task_results in evaluation.items():
        if not isinstance(task_results, list) or len(task_results) != 1:
            raise ValueError(f"expected greedy n=1 result for {task_id} in {path}")
        result = task_results[0]
        base_passed = result.get("base_status") == "pass"
        extra_passed = result.get("plus_status") == "pass"
        outcomes[str(task_id)] = {
            "base_passed": base_passed,
            "plus_passed": base_passed and extra_passed,
        }
    return outcomes


def compare_dataset(
    base_path: Path,
    final_path: Path,
    *,
    expected_tasks: int,
) -> dict[str, Any]:
    base = load_greedy_evalplus_outcomes(base_path)
    final = load_greedy_evalplus_outcomes(final_path)
    base_tasks = set(base)
    final_tasks = set(final)
    if len(base_tasks) != expected_tasks:
        raise ValueError(f"base has {len(base_tasks)} tasks; expected {expected_tasks}")
    if final_tasks != base_tasks:
        missing = sorted(base_tasks - final_tasks)
        extra = sorted(final_tasks - base_tasks)
        raise ValueError(f"final task set differs: missing={missing}, extra={extra}")

    base_plus = {task_id: value["plus_passed"] for task_id, value in base.items()}
    final_plus = {task_id: value["plus_passed"] for task_id, value in final.items()}
    base_plus_passed = sum(base_plus.values())
    final_plus_passed = sum(final_plus.values())
    gained = sum(not base_plus[task_id] and final_plus[task_id] for task_id in base_tasks)
    lost = sum(base_plus[task_id] and not final_plus[task_id] for task_id in base_tasks)
    return {
        "expected_tasks": expected_tasks,
        "base": {
            "path": str(base_path),
            "sha256": sha256_file(base_path),
            "base_tests_passed": sum(value["base_passed"] for value in base.values()),
            "plus_passed": base_plus_passed,
            "plus_pass_at_1": base_plus_passed / expected_tasks,
        },
        "final": {
            "path": str(final_path),
            "sha256": sha256_file(final_path),
            "base_tests_passed": sum(value["base_passed"] for value in final.values()),
            "plus_passed": final_plus_passed,
            "plus_pass_at_1": final_plus_passed / expected_tasks,
        },
        "paired_plus_transitions": {
            "gained_tasks": gained,
            "lost_tasks": lost,
            "net_tasks": gained - lost,
        },
        "no_regression": final_plus_passed >= base_plus_passed,
    }


def build_guardrail_report(
    base_humaneval: Path,
    final_humaneval: Path,
    base_mbpp: Path,
    final_mbpp: Path,
    *,
    humaneval_tasks: int = 164,
    mbpp_tasks: int = 378,
) -> dict[str, Any]:
    datasets = {
        "humaneval": compare_dataset(
            base_humaneval,
            final_humaneval,
            expected_tasks=humaneval_tasks,
        ),
        "mbpp": compare_dataset(base_mbpp, final_mbpp, expected_tasks=mbpp_tasks),
    }
    return {
        "protocol": "official EvalPlus greedy n=1; Plus means base and extra tests pass",
        "datasets": datasets,
        "evalplus_no_regression_passed": all(
            dataset["no_regression"] for dataset in datasets.values()
        ),
        "claim_boundary": (
            "Final guardrail only. These results must not select a training checkpoint or scale."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-humaneval", type=Path, required=True)
    parser.add_argument("--final-humaneval", type=Path, required=True)
    parser.add_argument("--base-mbpp", type=Path, required=True)
    parser.add_argument("--final-mbpp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--humaneval-tasks", type=int, default=164)
    parser.add_argument("--mbpp-tasks", type=int, default=378)
    args = parser.parse_args()
    report = build_guardrail_report(
        args.base_humaneval,
        args.final_humaneval,
        args.base_mbpp,
        args.final_mbpp,
        humaneval_tasks=args.humaneval_tasks,
        mbpp_tasks=args.mbpp_tasks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["evalplus_no_regression_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
