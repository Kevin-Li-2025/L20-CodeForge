#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from l20_codeforge.data.code_rlvr import (
    jaccard_similarity,
    normalize_prompt,
    prompt_shingles,
    sha256_file,
    stable_hash,
)


def load_task_prompts(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(
                    {
                        "source": str(path),
                        "id": str(row.get("task_id") or row.get("question_id") or len(rows)),
                        "prompt": str(row.get("prompt") or row.get("question") or ""),
                    }
                )
    return rows


def load_lcb_prompts(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"expected a list in {path}")
        for row in payload:
            prompt = str(row.get("prompt") or row.get("question_content") or "")
            digest = stable_hash(normalize_prompt(prompt))
            if not prompt or digest in seen:
                continue
            seen.add(digest)
            rows.append(
                {
                    "source": str(path),
                    "id": str(row.get("question_id") or row.get("task_id") or len(rows)),
                    "prompt": prompt,
                }
            )
    return rows


def audit_overlap(
    tasks: list[dict[str, str]], references: list[dict[str, str]], threshold: float
) -> dict[str, Any]:
    reference_hashes = {stable_hash(normalize_prompt(row["prompt"])): row for row in references}
    reference_shingles = [(row, prompt_shingles(row["prompt"])) for row in references]
    exact: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    top_matches: list[dict[str, Any]] = []
    for task in tasks:
        digest = stable_hash(normalize_prompt(task["prompt"]))
        if digest in reference_hashes:
            exact.append({"task": task, "reference": reference_hashes[digest]})
        task_shingles = prompt_shingles(task["prompt"])
        best_score = 0.0
        best_reference: dict[str, str] | None = None
        for reference, shingles in reference_shingles:
            score = jaccard_similarity(task_shingles, shingles)
            if score > best_score:
                best_score = score
                best_reference = reference
        match = {
            "task_id": task["id"],
            "task_source": task["source"],
            "reference_id": best_reference["id"] if best_reference else None,
            "reference_source": best_reference["source"] if best_reference else None,
            "five_gram_jaccard": round(best_score, 6),
        }
        top_matches.append(match)
        if best_score >= threshold:
            near.append(match)
    top_matches.sort(key=lambda row: row["five_gram_jaccard"], reverse=True)
    return {
        "tasks": len(tasks),
        "references": len(references),
        "threshold": threshold,
        "exact_matches": exact,
        "near_matches": near,
        "top_matches": top_matches[:20],
        "status": "PASS" if not exact and not near else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, action="append", required=True)
    parser.add_argument("--lcb-generations", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--fail-on-overlap", action="store_true")
    args = parser.parse_args()

    report = audit_overlap(
        load_task_prompts(args.tasks),
        load_lcb_prompts(args.lcb_generations),
        threshold=args.threshold,
    )
    report["task_sha256"] = {str(path): sha256_file(path) for path in args.tasks}
    report["reference_sha256"] = {str(path): sha256_file(path) for path in args.lcb_generations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.fail_on_overlap and report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
