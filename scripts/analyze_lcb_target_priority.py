#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lcb_behavior_test_prompts import (  # noqa: E402
    public_signal_features,
    select_target_records,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_selection_records(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("selection payload must be a list or contain records")
    return [record for record in records if isinstance(record, dict)]


def load_hidden_pass_by_id(path: Path) -> dict[str, bool]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError("eval_all payload must be a list")
    pass_by_id: dict[str, bool] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        question_id = record.get("question_id")
        graded = record.get("graded_list")
        if isinstance(question_id, str):
            pass_by_id[question_id] = bool(graded and graded[0] is True)
    return pass_by_id


def distribution(values: list[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def summarize_targets(
    records: list[dict[str, Any]],
    hidden_pass_by_id: dict[str, bool],
    budget: int,
) -> dict[str, Any]:
    selected = records[:budget]
    hidden_known = [
        record
        for record in selected
        if isinstance(record.get("question_id"), str)
        and record["question_id"] in hidden_pass_by_id
    ]
    hidden_fail_count = sum(
        not hidden_pass_by_id[record["question_id"]] for record in hidden_known
    )
    feature_rows = [public_signal_features(record) for record in selected]
    return {
        "budget": budget,
        "selected": len(selected),
        "hidden_known": len(hidden_known),
        "hidden_fail_count": hidden_fail_count,
        "hidden_fail_rate": round(hidden_fail_count / len(hidden_known), 6)
        if hidden_known
        else None,
        "public_pass_count_distribution": distribution(
            [int(features["public_pass_count"]) for features in feature_rows]
        ),
        "partial_public_score_count_distribution": distribution(
            [int(features["partial_public_score_count"]) for features in feature_rows]
        ),
        "first_question_ids": [record.get("question_id") for record in selected[:12]],
    }


def render_readme(summary: dict[str, Any]) -> str:
    lines = [
        "# LCB Target Priority Analysis",
        "",
        "This analysis compares public-signal-only targeting policies for the next "
        "candidate-aware behavior-test generation batch. Hidden-test outcomes are "
        "used only after the fact to measure target density; they are not inputs to "
        "the prompt builder or selector.",
        "",
        "## Budget Comparison",
        "",
        "| priority | budget | selected | hidden fail count | hidden fail rate | public pass count distribution |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for priority in summary["priorities"]:
        for item in priority["budgets"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        priority["target_priority"],
                        str(item["budget"]),
                        str(item["selected"]),
                        str(item["hidden_fail_count"]),
                        f"{item['hidden_fail_rate']:.4f}"
                        if item["hidden_fail_rate"] is not None
                        else "n/a",
                        "`" + json.dumps(item["public_pass_count_distribution"]) + "`",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "`public-fragility` is the default next-batch targeter: it still uses only "
            "public scores, but prioritizes public-passing ties with fewer "
            "public-passing candidates and more partial public failures. That moves "
            "the behavior-test budget away from easy all-candidates-pass tasks and "
            "toward cases where candidate-aware verification can plausibly change "
            "the selected solution.",
            "",
        ]
    )
    return "\n".join(lines)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    records = load_selection_records(Path(args.public_selection))
    hidden_pass_by_id = load_hidden_pass_by_id(Path(args.eval_all))
    priority_summaries = []
    for priority in args.priorities:
        selected = select_target_records(
            records,
            min_public_score=args.min_public_score,
            include_non_ties=args.include_non_ties,
            limit=None,
            target_priority=priority,
        )
        priority_summaries.append(
            {
                "target_priority": priority,
                "eligible_count": len(selected),
                "budgets": [
                    summarize_targets(selected, hidden_pass_by_id, budget)
                    for budget in args.budgets
                    if budget <= len(selected)
                ],
            }
        )
    return {
        "schema_version": 1,
        "purpose": "public-signal target-priority analysis for LCB behavior-test generation",
        "leakage_policy": (
            "Hidden outcomes are used only for this retrospective density analysis, "
            "not for prompt construction or candidate selection."
        ),
        "inputs": {
            "public_selection": args.public_selection,
            "eval_all": args.eval_all,
            "min_public_score": args.min_public_score,
            "include_non_ties": args.include_non_ties,
            "budgets": args.budgets,
        },
        "priorities": priority_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze public-only target priorities for LCB behavior-test generation."
    )
    parser.add_argument("--public-selection", required=True)
    parser.add_argument("--eval-all", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--priorities",
        nargs="+",
        choices=["input-order", "public-ambiguity", "public-fragility"],
        default=["input-order", "public-fragility", "public-ambiguity"],
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[32, 54, 64, 96, 128, 192, 256],
    )
    parser.add_argument("--min-public-score", type=float, default=1.0)
    parser.add_argument("--include-non-ties", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
