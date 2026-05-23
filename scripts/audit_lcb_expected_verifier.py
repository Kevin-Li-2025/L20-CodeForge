#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_lcb_selection_runs import load_eval_passes, load_selection_records  # noqa: E402
from select_lcb_expected_verifier_candidates import (  # noqa: E402
    build_expected_verifier_selection,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def option_maps_by_question(
    candidate_outputs_payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    maps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidate_outputs_payload.get("candidate_outputs", []):
        question_id = record.get("question_id")
        if not isinstance(question_id, str):
            continue
        maps[question_id].append(
            {
                "input_index": record.get("input_index"),
                "input_sha256_16": record.get("input_sha256_16"),
                "options": [
                    {
                        "label": option.get("label"),
                        "candidate_indices": option.get("candidate_indices", []),
                    }
                    for option in record.get("options", [])
                    if isinstance(option, dict)
                ],
            }
        )
    return maps


def choice_votes_by_question(
    verifier_choices_payload: dict[str, Any],
) -> dict[str, dict[str, dict[str, float | int]]]:
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    confidence: dict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for record in verifier_choices_payload.get("records", []):
        question_id = record.get("question_id")
        if not isinstance(question_id, str):
            continue
        label = str(record.get("choice") or "").upper()
        votes[question_id][label] += 1
        confidence[question_id][label] += float(record.get("confidence", 0.0))
    return {
        question_id: {
            label: {
                "choice_count": count,
                "confidence_sum": confidence[question_id][label],
            }
            for label, count in sorted(counter.items())
        }
        for question_id, counter in votes.items()
    }


def stabilized_candidate_passes(
    candidate_eval: dict[str, dict[str, Any]],
    comparison_summary: dict[str, Any],
) -> dict[str, bool]:
    passes = {
        question_id: bool(record["passed"])
        for question_id, record in candidate_eval.items()
    }
    for record in comparison_summary.get("changed_or_outcome_different", []):
        question_id = record.get("question_id")
        if isinstance(question_id, str) and "stabilized_candidate_pass" in record:
            passes[question_id] = bool(record["stabilized_candidate_pass"])
    return passes


def override_audit_records(
    comparison_summary: dict[str, Any],
    candidate_selection: dict[str, dict[str, Any]],
    verifier_choices_payload: dict[str, Any],
    candidate_outputs_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    votes_by_question = choice_votes_by_question(verifier_choices_payload)
    maps_by_question = option_maps_by_question(candidate_outputs_payload)
    records = []
    for comparison in comparison_summary.get("changed_or_outcome_different", []):
        if not comparison.get("selection_changed"):
            continue
        question_id = comparison.get("question_id")
        if not isinstance(question_id, str):
            continue
        selection = candidate_selection.get(question_id, {})
        records.append(
            {
                "question_id": question_id,
                "question_title": comparison.get("question_title"),
                "baseline_selected_index": comparison.get("baseline_selected_index"),
                "candidate_selected_index": comparison.get("candidate_selected_index"),
                "baseline_pass": comparison.get("baseline_pass"),
                "candidate_pass": comparison.get("candidate_pass"),
                "stabilized_candidate_pass": comparison.get(
                    "stabilized_candidate_pass"
                ),
                "verifier_selected_index": selection.get("verifier_selected_index"),
                "verifier_selected_choice_count": selection.get(
                    "verifier_selected_choice_count"
                ),
                "verifier_confidence_margin_vs_public": selection.get(
                    "verifier_confidence_margin_vs_public"
                ),
                "expected_verifier_scores": selection.get("expected_verifier_scores"),
                "choice_votes": votes_by_question.get(question_id, {}),
                "option_maps": maps_by_question.get(question_id, []),
            }
        )
    return records


def infer_threshold_sweep(
    public_selection_payload: dict[str, Any],
    candidate_outputs_payload: dict[str, Any],
    verifier_choices_payload: dict[str, Any],
    baseline_eval: dict[str, dict[str, Any]],
    candidate_eval: dict[str, dict[str, Any]],
    baseline_selection: dict[str, dict[str, Any]],
    current_candidate_selection: dict[str, dict[str, Any]],
    stabilized_candidate_eval: dict[str, bool],
    min_choice_counts: list[int],
    min_confidence_margins: list[float],
) -> list[dict[str, Any]]:
    target_question_ids = sorted(set(baseline_eval) & set(candidate_eval))
    baseline_passed = sum(
        bool(baseline_eval[question_id]["passed"]) for question_id in target_question_ids
    )
    sweep = []
    for min_choice_count in min_choice_counts:
        for min_confidence_margin in min_confidence_margins:
            selection_payload = build_expected_verifier_selection(
                public_selection_payload=public_selection_payload,
                candidate_outputs_payload=candidate_outputs_payload,
                verifier_choices_payload=verifier_choices_payload,
                min_choice_count=min_choice_count,
                min_confidence_margin=min_confidence_margin,
            )
            selection_by_id = {
                record["question_id"]: record for record in selection_payload["records"]
            }
            raw_passed = 0
            stabilized_passed = 0
            target_override_question_ids = []
            unknown_question_ids = []
            for question_id in target_question_ids:
                selected_index = selection_by_id[question_id]["selected_index"]
                baseline_index = baseline_selection[question_id]["selected_index"]
                current_index = current_candidate_selection[question_id][
                    "selected_index"
                ]
                if selected_index != baseline_index:
                    target_override_question_ids.append(question_id)
                if selected_index == current_index:
                    raw_passed += int(candidate_eval[question_id]["passed"])
                    stabilized_passed += int(stabilized_candidate_eval[question_id])
                elif selected_index == baseline_index:
                    raw_passed += int(baseline_eval[question_id]["passed"])
                    stabilized_passed += int(baseline_eval[question_id]["passed"])
                else:
                    unknown_question_ids.append(question_id)
            sweep.append(
                {
                    "min_choice_count": min_choice_count,
                    "min_confidence_margin": min_confidence_margin,
                    "global_override_count": selection_payload["metadata"][
                        "override_count"
                    ],
                    "target_override_count": len(target_override_question_ids),
                    "target_override_question_ids": target_override_question_ids,
                    "inferred_raw_passed": raw_passed,
                    "inferred_raw_net_gain": raw_passed - baseline_passed,
                    "inferred_stabilized_passed": stabilized_passed,
                    "inferred_stabilized_net_gain": (
                        stabilized_passed - baseline_passed
                    ),
                    "unknown_question_ids": unknown_question_ids,
                }
            )
    return sweep


def build_audit(
    comparison_summary: dict[str, Any],
    public_selection_payload: dict[str, Any],
    candidate_outputs_payload: dict[str, Any],
    verifier_choices_payload: dict[str, Any],
    baseline_eval: dict[str, dict[str, Any]],
    candidate_eval: dict[str, dict[str, Any]],
    baseline_selection: dict[str, dict[str, Any]],
    current_candidate_selection: dict[str, dict[str, Any]],
    min_choice_counts: list[int],
    min_confidence_margins: list[float],
) -> dict[str, Any]:
    stabilized_passes = stabilized_candidate_passes(candidate_eval, comparison_summary)
    return {
        "metadata": {
            "source": "lcb_expected_output_verifier_audit_v1",
            "sweep_inference": (
                "post_hoc_from_existing_baseline_and_candidate_hidden_replays"
            ),
        },
        "comparison_summary": {
            key: value
            for key, value in comparison_summary.items()
            if key != "changed_or_outcome_different"
        },
        "override_audit_records": override_audit_records(
            comparison_summary=comparison_summary,
            candidate_selection=current_candidate_selection,
            verifier_choices_payload=verifier_choices_payload,
            candidate_outputs_payload=candidate_outputs_payload,
        ),
        "threshold_sweep": infer_threshold_sweep(
            public_selection_payload=public_selection_payload,
            candidate_outputs_payload=candidate_outputs_payload,
            verifier_choices_payload=verifier_choices_payload,
            baseline_eval=baseline_eval,
            candidate_eval=candidate_eval,
            baseline_selection=baseline_selection,
            current_candidate_selection=current_candidate_selection,
            stabilized_candidate_eval=stabilized_passes,
            min_choice_counts=min_choice_counts,
            min_confidence_margins=min_confidence_margins,
        ),
    }


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit expected-output verifier overrides and threshold sweeps."
    )
    parser.add_argument("--comparison-summary", required=True)
    parser.add_argument("--public-selection", required=True)
    parser.add_argument("--candidate-outputs", required=True)
    parser.add_argument("--verifier-choices", required=True)
    parser.add_argument("--candidate-selection", required=True)
    parser.add_argument("--baseline-eval-all", required=True)
    parser.add_argument("--candidate-eval-all", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-choice-counts", default="2,3,4,5,6,7,8")
    parser.add_argument("--min-confidence-margins", default="1,2,3,4,5,6,7,8")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    public_selection_payload = load_json(Path(args.public_selection))
    audit = build_audit(
        comparison_summary=load_json(Path(args.comparison_summary)),
        public_selection_payload=public_selection_payload,
        candidate_outputs_payload=load_json(Path(args.candidate_outputs)),
        verifier_choices_payload=load_json(Path(args.verifier_choices)),
        baseline_eval=load_eval_passes(Path(args.baseline_eval_all)),
        candidate_eval=load_eval_passes(Path(args.candidate_eval_all)),
        baseline_selection=load_selection_records(Path(args.public_selection)),
        current_candidate_selection=load_selection_records(Path(args.candidate_selection)),
        min_choice_counts=parse_int_list(args.min_choice_counts),
        min_confidence_margins=parse_float_list(args.min_confidence_margins),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(audit["comparison_summary"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
