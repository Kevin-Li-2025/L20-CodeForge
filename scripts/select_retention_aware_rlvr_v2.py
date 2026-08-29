#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
            if line.strip():
                row = json.loads(line)
                grouped.setdefault(str(row["task_id"]), []).append(row)
    if not grouped:
        raise ValueError(f"no rollout outcomes found in {path}")
    return {
        task_id: bool(min(rows, key=lambda row: int(row.get("sample_index", 0)))["all_passed"])
        for task_id, rows in grouped.items()
    }


def _score(path: Path, expected_tasks: set[str], label: str) -> dict[str, Any]:
    outcomes = load_greedy_outcomes(path)
    task_ids = set(outcomes)
    if task_ids != expected_tasks:
        missing = sorted(expected_tasks - task_ids)
        extra = sorted(task_ids - expected_tasks)
        raise ValueError(f"{label} task set differs: missing={missing}, extra={extra}")
    passed = sum(outcomes.values())
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "tasks": len(outcomes),
        "passed": passed,
        "greedy_accuracy": passed / len(outcomes),
    }


def build_selection_report(
    base_target_path: Path,
    base_retention_path: Path,
    candidates: dict[str, tuple[Path, Path]],
    *,
    expected_target_tasks: int = 200,
    expected_retention_tasks: int = 90,
    required_absolute_delta: float = 0.02,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one RLVR candidate is required")
    base_target_outcomes = load_greedy_outcomes(base_target_path)
    base_retention_outcomes = load_greedy_outcomes(base_retention_path)
    if len(base_target_outcomes) != expected_target_tasks:
        raise ValueError(
            f"base target has {len(base_target_outcomes)} tasks; expected {expected_target_tasks}"
        )
    if len(base_retention_outcomes) != expected_retention_tasks:
        raise ValueError(
            "base retention has "
            f"{len(base_retention_outcomes)} tasks; expected {expected_retention_tasks}"
        )
    target_ids = set(base_target_outcomes)
    retention_ids = set(base_retention_outcomes)
    base_target = _score(base_target_path, target_ids, "base target")
    base_retention = _score(base_retention_path, retention_ids, "base retention")
    required_target_passes = base_target["passed"] + math.ceil(
        required_absolute_delta * expected_target_tasks - 1e-12
    )

    rows: list[dict[str, Any]] = []
    for name, (target_path, retention_path) in sorted(candidates.items()):
        target = _score(target_path, target_ids, f"{name} target")
        retention = _score(retention_path, retention_ids, f"{name} retention")
        target_delta = target["passed"] - base_target["passed"]
        retention_delta = retention["passed"] - base_retention["passed"]
        rows.append(
            {
                "name": name,
                "target": target,
                "retention": retention,
                "target_pass_delta_vs_base": target_delta,
                "target_absolute_delta_vs_base": target_delta / expected_target_tasks,
                "retention_pass_delta_vs_base": retention_delta,
                "target_gate_passed": target["passed"] >= required_target_passes,
                "retention_gate_passed": retention["passed"] >= base_retention["passed"],
                "eligible": target["passed"] >= required_target_passes
                and retention["passed"] >= base_retention["passed"],
            }
        )

    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -int(row["target"]["passed"]),
            -int(row["retention"]["passed"]),
            str(row["name"]),
        )
    )
    selected = eligible[0] if eligible else None
    return {
        "protocol": (
            "greedy n=1 on frozen new rStar development and MBPP official validation; "
            "require at least +2.0 target points and no MBPP-validation regression; then "
            "maximize target passes, break ties by retention passes and lexical seed name"
        ),
        "base_target": base_target,
        "base_retention": base_retention,
        "required_absolute_delta": required_absolute_delta,
        "required_target_passes": required_target_passes,
        "candidates": rows,
        "selected": selected,
        "development_gate_passed": selected is not None,
        "claim_boundary": (
            "Development selection only. New rStar final, date-held-out LiveCodeBench, and "
            "EvalPlus are not read by this selector."
        ),
    }


def parse_candidate(value: str) -> tuple[str, tuple[Path, Path]]:
    name, separator, paths = value.partition("=")
    target, path_separator, retention = paths.partition(",")
    if not separator or not path_separator or not name or not target or not retention:
        raise argparse.ArgumentTypeError(
            "candidate must have the form NAME=TARGET_JSONL,RETENTION_JSONL"
        )
    return name, (Path(target), Path(retention))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-target", type=Path, required=True)
    parser.add_argument("--base-retention", type=Path, required=True)
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-target-tasks", type=int, default=200)
    parser.add_argument("--expected-retention-tasks", type=int, default=90)
    parser.add_argument("--required-absolute-delta", type=float, default=0.02)
    args = parser.parse_args()
    candidate_pairs: list[tuple[str, tuple[Path, Path]]] = args.candidate
    candidates = dict(candidate_pairs)
    if len(candidates) != len(candidate_pairs):
        parser.error("candidate names must be unique")
    report = build_selection_report(
        args.base_target,
        args.base_retention,
        candidates,
        expected_target_tasks=args.expected_target_tasks,
        expected_retention_tasks=args.expected_retention_tasks,
        required_absolute_delta=args.required_absolute_delta,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["development_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
