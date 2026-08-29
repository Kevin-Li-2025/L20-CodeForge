#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_greedy_outcomes(path: Path) -> dict[str, bool]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped.setdefault(str(row["task_id"]), []).append(row)
    return {
        task_id: bool(min(rows, key=lambda row: int(row.get("sample_index", 0)))["all_passed"])
        for task_id, rows in grouped.items()
    }


def exact_mcnemar_p_value(gained: int, lost: int) -> float:
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gained, lost) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_bootstrap_interval(
    before: list[bool],
    after: list[bool],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> list[float]:
    rng = random.Random(seed)
    differences = [int(right) - int(left) for left, right in zip(before, after, strict=True)]
    size = len(differences)
    draws = sorted(
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    lower = draws[int(0.025 * samples)]
    upper = draws[min(samples - 1, int(0.975 * samples))]
    return [lower, upper]


def compare_pair(
    before: dict[str, bool],
    after: dict[str, bool],
    task_ids: list[str],
) -> dict[str, Any]:
    before_values = [before[task_id] for task_id in task_ids]
    after_values = [after[task_id] for task_id in task_ids]
    gained = sum(not left and right for left, right in zip(before_values, after_values, strict=True))
    lost = sum(left and not right for left, right in zip(before_values, after_values, strict=True))
    delta = (gained - lost) / len(task_ids)
    return {
        "gained_tasks": gained,
        "lost_tasks": lost,
        "net_tasks": gained - lost,
        "absolute_delta": delta,
        "percentage_point_delta": 100.0 * delta,
        "paired_bootstrap_95pct_absolute_delta": paired_bootstrap_interval(
            before_values, after_values
        ),
        "exact_mcnemar_p_value": exact_mcnemar_p_value(gained, lost),
    }


def build_report(
    stage_paths: dict[str, Path],
    *,
    expected_tasks: int,
    required_base_to_final_delta: float,
) -> dict[str, Any]:
    outcomes = {name: load_greedy_outcomes(path) for name, path in stage_paths.items()}
    task_sets = {name: set(values) for name, values in outcomes.items()}
    reference_tasks = task_sets["base"]
    if len(reference_tasks) != expected_tasks:
        raise ValueError(f"base has {len(reference_tasks)} tasks; expected {expected_tasks}")
    mismatches = {
        name: {
            "missing": sorted(reference_tasks - task_ids),
            "extra": sorted(task_ids - reference_tasks),
        }
        for name, task_ids in task_sets.items()
        if task_ids != reference_tasks
    }
    if mismatches:
        raise ValueError(f"stage task sets differ: {mismatches}")

    task_ids = sorted(reference_tasks)
    stages = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "tasks": len(task_ids),
            "passed": sum(values.values()),
            "greedy_accuracy": sum(values.values()) / len(task_ids),
        }
        for (name, path), values in zip(stage_paths.items(), outcomes.values(), strict=True)
    }
    pairs = {
        "base_to_sft": compare_pair(outcomes["base"], outcomes["sft"], task_ids),
        "sft_to_rlvr": compare_pair(outcomes["sft"], outcomes["rlvr"], task_ids),
        "base_to_rlvr": compare_pair(outcomes["base"], outcomes["rlvr"], task_ids),
    }
    return {
        "protocol": "same frozen dev tasks; greedy n=1; sample_index=0",
        "expected_tasks": expected_tasks,
        "stages": stages,
        "paired_comparisons": pairs,
        "development_gate": {
            "required_base_to_final_absolute_delta": required_base_to_final_delta,
            "observed_base_to_final_absolute_delta": pairs["base_to_rlvr"]["absolute_delta"],
            "passed": pairs["base_to_rlvr"]["absolute_delta"]
            >= required_base_to_final_delta,
        },
        "claim_boundary": (
            "Development-set comparison only. EvalPlus is a separate final guardrail and is not "
            "consumed by this report."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--rlvr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=200)
    parser.add_argument("--required-base-to-final-delta", type=float, default=0.02)
    args = parser.parse_args()
    report = build_report(
        {"base": args.base, "sft": args.sft, "rlvr": args.rlvr},
        expected_tasks=args.expected_tasks,
        required_base_to_final_delta=args.required_base_to_final_delta,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
