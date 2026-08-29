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
            if not line.strip():
                continue
            row = json.loads(line)
            grouped.setdefault(str(row["task_id"]), []).append(row)
    return {
        task_id: bool(min(rows, key=lambda row: int(row.get("sample_index", 0)))["all_passed"])
        for task_id, rows in grouped.items()
    }


def build_selection_report(
    base_path: Path,
    candidates: dict[float, Path],
    *,
    expected_tasks: int,
    required_absolute_delta: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    if any(not 0.0 < scale <= 1.0 for scale in candidates):
        raise ValueError("candidate scales must satisfy 0 < scale <= 1")

    base = load_greedy_outcomes(base_path)
    base_tasks = set(base)
    if len(base_tasks) != expected_tasks:
        raise ValueError(f"base has {len(base_tasks)} tasks; expected {expected_tasks}")
    base_passed = sum(base.values())
    required_passed = base_passed + math.ceil(required_absolute_delta * expected_tasks - 1e-12)

    rows: list[dict[str, Any]] = []
    for scale, path in sorted(candidates.items()):
        outcomes = load_greedy_outcomes(path)
        task_ids = set(outcomes)
        if task_ids != base_tasks:
            missing = sorted(base_tasks - task_ids)
            extra = sorted(task_ids - base_tasks)
            raise ValueError(f"scale {scale} task set differs: missing={missing}, extra={extra}")
        passed = sum(outcomes.values())
        rows.append(
            {
                "scale": scale,
                "path": str(path),
                "sha256": sha256_file(path),
                "tasks": len(outcomes),
                "passed": passed,
                "greedy_accuracy": passed / expected_tasks,
                "absolute_delta_vs_base": (passed - base_passed) / expected_tasks,
                "eligible": passed >= required_passed,
            }
        )

    selected = next((row for row in rows if row["eligible"]), None)
    return {
        "protocol": (
            "same frozen development tasks; greedy n=1; select the smallest adapter scale "
            "meeting the predeclared Base-to-final improvement gate"
        ),
        "base": {
            "path": str(base_path),
            "sha256": sha256_file(base_path),
            "tasks": len(base),
            "passed": base_passed,
            "greedy_accuracy": base_passed / expected_tasks,
        },
        "required_absolute_delta": required_absolute_delta,
        "required_passed": required_passed,
        "candidates": rows,
        "selected": selected,
        "development_gate_passed": selected is not None,
        "claim_boundary": (
            "Development data only. EvalPlus scores are neither read nor used for scale selection."
        ),
    }


def parse_candidate(value: str) -> tuple[float, Path]:
    scale_text, separator, path_text = value.partition("=")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError("candidate must have the form SCALE=PATH")
    try:
        scale = float(scale_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid scale: {scale_text}") from error
    return scale, Path(path_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=200)
    parser.add_argument("--required-absolute-delta", type=float, default=0.02)
    args = parser.parse_args()
    candidate_pairs: list[tuple[float, Path]] = args.candidate
    candidates = dict(candidate_pairs)
    if len(candidates) != len(candidate_pairs):
        parser.error("candidate scales must be unique")
    report = build_selection_report(
        args.base,
        candidates,
        expected_tasks=args.expected_tasks,
        required_absolute_delta=args.required_absolute_delta,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["development_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
