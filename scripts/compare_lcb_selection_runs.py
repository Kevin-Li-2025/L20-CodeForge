#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_passes(path: Path) -> dict[str, dict[str, Any]]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError("eval_all payload must be a list")
    passes: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        question_id = record.get("question_id")
        if not isinstance(question_id, str):
            continue
        graded = record.get("graded_list")
        passes[question_id] = {
            "question_id": question_id,
            "question_title": record.get("question_title"),
            "passed": bool(graded and graded[0] is True),
        }
    return passes


def load_selection_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("selection payload must be a list or contain records")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        question_id = record.get("question_id")
        if isinstance(question_id, str):
            by_id[question_id] = record
    return by_id


def selected_index(record: dict[str, Any] | None) -> int | None:
    if not record:
        return None
    value = record.get("selected_index")
    return int(value) if isinstance(value, int) else None


def selected_public_score(record: dict[str, Any] | None) -> float | None:
    if not record:
        return None
    value = record.get("selected_public_score")
    return float(value) if isinstance(value, (int, float)) else None


def summarize_comparison(
    baseline_eval: dict[str, dict[str, Any]],
    candidate_eval: dict[str, dict[str, Any]],
    baseline_selection: dict[str, dict[str, Any]],
    candidate_selection: dict[str, dict[str, Any]],
    candidate_recheck_eval: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate_recheck_eval = candidate_recheck_eval or {}
    question_ids = sorted(set(baseline_eval) & set(candidate_eval))
    changed_or_outcome_different = []
    unchanged_outcome_flips = []
    rechecked_unchanged_outcome_flips = []
    unresolved_unchanged_outcome_flips = []
    changed_selection_improvements = []
    changed_selection_regressions = []
    stabilized_candidate_pass_by_id: dict[str, bool] = {}

    for question_id in question_ids:
        baseline = baseline_eval[question_id]
        candidate = candidate_eval[question_id]
        baseline_record = baseline_selection.get(question_id)
        candidate_record = candidate_selection.get(question_id)
        baseline_index = selected_index(baseline_record)
        candidate_index = selected_index(candidate_record)
        selection_changed = baseline_index != candidate_index
        outcome_changed = baseline["passed"] != candidate["passed"]
        recheck_record = candidate_recheck_eval.get(question_id)
        candidate_recheck_pass = (
            bool(recheck_record["passed"]) if recheck_record is not None else None
        )
        stabilized_candidate_pass = candidate["passed"]
        if not selection_changed and outcome_changed:
            unchanged_outcome_flips.append(question_id)
            if candidate_recheck_pass is None:
                unresolved_unchanged_outcome_flips.append(question_id)
            else:
                rechecked_unchanged_outcome_flips.append(question_id)
                stabilized_candidate_pass = candidate_recheck_pass
                if stabilized_candidate_pass != baseline["passed"]:
                    unresolved_unchanged_outcome_flips.append(question_id)
        stabilized_candidate_pass_by_id[question_id] = stabilized_candidate_pass
        if selection_changed and (not baseline["passed"]) and candidate["passed"]:
            changed_selection_improvements.append(question_id)
        if selection_changed and baseline["passed"] and (not candidate["passed"]):
            changed_selection_regressions.append(question_id)
        if selection_changed or outcome_changed:
            changed_or_outcome_different.append(
                {
                    "question_id": question_id,
                    "question_title": candidate.get("question_title")
                    or baseline.get("question_title"),
                    "baseline_pass": baseline["passed"],
                    "candidate_pass": candidate["passed"],
                    "candidate_recheck_pass": candidate_recheck_pass,
                    "stabilized_candidate_pass": stabilized_candidate_pass,
                    "baseline_selected_index": baseline_index,
                    "candidate_selected_index": candidate_index,
                    "selection_changed": selection_changed,
                    "outcome_changed": outcome_changed,
                    "baseline_selected_public_score": selected_public_score(
                        baseline_record
                    ),
                    "candidate_selected_public_score": selected_public_score(
                        candidate_record
                    ),
                    "candidate_override_from_public": bool(
                        candidate_record
                        and candidate_record.get("override_from_public", False)
                    ),
                    "candidate_behavior_success_rate": (
                        candidate_record.get("behavior_success_rate")
                        if candidate_record
                        else None
                    ),
                    "candidate_behavior_consensus_margin_vs_public": (
                        candidate_record.get("behavior_consensus_margin_vs_public")
                        if candidate_record
                        else None
                    ),
                }
            )

    baseline_passed = sum(baseline_eval[question_id]["passed"] for question_id in question_ids)
    candidate_passed = sum(candidate_eval[question_id]["passed"] for question_id in question_ids)
    stabilized_candidate_passed = sum(
        stabilized_candidate_pass_by_id[question_id] for question_id in question_ids
    )
    if unchanged_outcome_flips:
        raw_status = "unstable_replay"
    elif changed_selection_regressions:
        raw_status = "regressed"
    elif candidate_passed > baseline_passed:
        raw_status = "improved"
    elif candidate_passed == baseline_passed:
        raw_status = "neutral"
    else:
        raw_status = "worse"

    if unresolved_unchanged_outcome_flips:
        status = "unstable_replay"
    elif changed_selection_regressions:
        status = "regressed"
    elif stabilized_candidate_passed > baseline_passed:
        status = "stabilized_improved" if unchanged_outcome_flips else "improved"
    elif stabilized_candidate_passed == baseline_passed:
        status = "stabilized_neutral" if unchanged_outcome_flips else "neutral"
    else:
        status = "stabilized_worse" if unchanged_outcome_flips else "worse"

    return {
        "status": status,
        "raw_status": raw_status,
        "total": len(question_ids),
        "baseline_passed": baseline_passed,
        "candidate_passed": candidate_passed,
        "net_gain": candidate_passed - baseline_passed,
        "stabilized_candidate_passed": stabilized_candidate_passed,
        "stabilized_net_gain": stabilized_candidate_passed - baseline_passed,
        "selection_changed_count": sum(
            item["selection_changed"] for item in changed_or_outcome_different
        ),
        "outcome_changed_count": sum(
            item["outcome_changed"] for item in changed_or_outcome_different
        ),
        "unchanged_outcome_flip_count": len(unchanged_outcome_flips),
        "rechecked_unchanged_outcome_flip_count": len(rechecked_unchanged_outcome_flips),
        "unresolved_unchanged_outcome_flip_count": len(unresolved_unchanged_outcome_flips),
        "changed_selection_improvement_count": len(changed_selection_improvements),
        "changed_selection_regression_count": len(changed_selection_regressions),
        "changed_or_outcome_different": changed_or_outcome_different,
        "unchanged_outcome_flips": unchanged_outcome_flips,
        "rechecked_unchanged_outcome_flips": rechecked_unchanged_outcome_flips,
        "unresolved_unchanged_outcome_flips": unresolved_unchanged_outcome_flips,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two LCB selection eval runs.")
    parser.add_argument("--baseline-eval-all", required=True)
    parser.add_argument("--candidate-eval-all", required=True)
    parser.add_argument("--baseline-selection", required=True)
    parser.add_argument("--candidate-selection", required=True)
    parser.add_argument(
        "--candidate-recheck-eval-all",
        action="append",
        default=[],
        help="Optional eval_all.json rechecks for candidate unchanged-outcome flips.",
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidate_rechecks: dict[str, dict[str, Any]] = {}
    for recheck_path in args.candidate_recheck_eval_all:
        candidate_rechecks.update(load_eval_passes(Path(recheck_path)))
    summary = summarize_comparison(
        baseline_eval=load_eval_passes(Path(args.baseline_eval_all)),
        candidate_eval=load_eval_passes(Path(args.candidate_eval_all)),
        baseline_selection=load_selection_records(Path(args.baseline_selection)),
        candidate_selection=load_selection_records(Path(args.candidate_selection)),
        candidate_recheck_eval=candidate_rechecks,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
