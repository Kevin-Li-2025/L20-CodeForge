#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lcb_expected_output_verifier_prompts import prompt_id  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def public_pass_indices(record: dict[str, Any]) -> list[int]:
    return [
        index
        for index, score in enumerate(record.get("public_scores", []))
        if float(score) == 1.0
    ]


def build_candidate_output_index(candidate_outputs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        prompt_id(record["question_id"], int(record["input_index"])): record
        for record in candidate_outputs
    }


def score_question_choices(
    choices: list[dict[str, Any]],
    candidate_outputs_by_record_id: dict[str, dict[str, Any]],
    allowed_candidate_indices: list[int],
) -> dict[int, dict[str, float | int]]:
    allowed = set(allowed_candidate_indices)
    scores: dict[int, dict[str, float | int]] = {
        index: {"confidence_sum": 0.0, "choice_count": 0}
        for index in allowed_candidate_indices
    }
    for choice in choices:
        label = str(choice.get("choice") or "").upper()
        if label == "NONE":
            continue
        candidate_output = candidate_outputs_by_record_id.get(str(choice.get("record_id")))
        if not candidate_output:
            continue
        options = {
            str(option.get("label")): option
            for option in candidate_output.get("options", [])
            if isinstance(option, dict)
        }
        option = options.get(label)
        if not option:
            continue
        confidence = float(choice.get("confidence", 0.0))
        for candidate_index in option.get("candidate_indices", []):
            if candidate_index in allowed:
                scores[candidate_index]["confidence_sum"] += confidence
                scores[candidate_index]["choice_count"] += 1
    return scores


def select_candidate_index(
    public_selected_index: int,
    allowed_candidate_indices: list[int],
    scores: dict[int, dict[str, float | int]],
    min_choice_count: int,
    min_confidence_margin: float,
) -> tuple[int, dict[str, Any]]:
    if not allowed_candidate_indices:
        return public_selected_index, {"override_allowed": False}
    public_score = float(scores.get(public_selected_index, {}).get("confidence_sum", 0.0))
    ranked = sorted(
        allowed_candidate_indices,
        key=lambda index: (
            -float(scores.get(index, {}).get("confidence_sum", 0.0)),
            -int(scores.get(index, {}).get("choice_count", 0)),
            index != public_selected_index,
            index,
        ),
    )
    selected_index = ranked[0]
    selected_score = float(scores.get(selected_index, {}).get("confidence_sum", 0.0))
    selected_count = int(scores.get(selected_index, {}).get("choice_count", 0))
    margin = selected_score - public_score
    override_allowed = (
        selected_index != public_selected_index
        and selected_count >= min_choice_count
        and margin >= min_confidence_margin
    )
    if not override_allowed:
        selected_index = public_selected_index
    return selected_index, {
        "verifier_selected_index": ranked[0],
        "verifier_selected_confidence_sum": selected_score,
        "verifier_selected_choice_count": selected_count,
        "verifier_public_confidence_sum": public_score,
        "verifier_confidence_margin_vs_public": margin,
        "override_allowed": override_allowed,
    }


def build_expected_verifier_selection(
    public_selection_payload: dict[str, Any],
    candidate_outputs_payload: dict[str, Any],
    verifier_choices_payload: dict[str, Any],
    min_choice_count: int,
    min_confidence_margin: float,
) -> dict[str, Any]:
    candidate_outputs_by_record_id = build_candidate_output_index(
        candidate_outputs_payload.get("candidate_outputs", [])
    )
    choices_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for choice in verifier_choices_payload.get("records", []):
        question_id = choice.get("question_id")
        if isinstance(question_id, str):
            choices_by_question[question_id].append(choice)

    records = []
    for record in public_selection_payload.get("records", []):
        question_id = record.get("question_id")
        if not isinstance(question_id, str):
            continue
        public_selected_index = int(record["selected_index"])
        allowed_indices = public_pass_indices(record)
        scores = score_question_choices(
            choices_by_question.get(question_id, []),
            candidate_outputs_by_record_id,
            allowed_indices,
        )
        selected_index, verifier_features = select_candidate_index(
            public_selected_index,
            allowed_indices,
            scores,
            min_choice_count=min_choice_count,
            min_confidence_margin=min_confidence_margin,
        )
        output_record = dict(record)
        output_record.update(
            {
                "selected_index": selected_index,
                "selected_public_score": (
                    float(record.get("public_scores", [0.0])[selected_index])
                    if selected_index < len(record.get("public_scores", []))
                    else 0.0
                ),
                "expected_verifier_scores": scores,
                "expected_verifier_choice_count": len(choices_by_question.get(question_id, [])),
                "override_from_public": selected_index != public_selected_index,
                **verifier_features,
            }
        )
        records.append(output_record)

    return {
        "metrics": public_selection_payload.get("metrics"),
        "records": records,
        "metadata": {
            "source": "lcb_expected_output_choice_verifier_v1",
            "candidate_outputs_metadata": candidate_outputs_payload.get("metadata"),
            "verifier_choices_metadata": verifier_choices_payload.get("metadata"),
            "min_choice_count": min_choice_count,
            "min_confidence_margin": min_confidence_margin,
            "override_count": sum(record["override_from_public"] for record in records),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select LCB candidates using expected-output verifier choices."
    )
    parser.add_argument("--public-selection", required=True)
    parser.add_argument("--candidate-outputs", required=True)
    parser.add_argument("--verifier-choices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-choice-count", type=int, default=2)
    parser.add_argument("--min-confidence-margin", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_expected_verifier_selection(
        public_selection_payload=load_json(Path(args.public_selection)),
        candidate_outputs_payload=load_json(Path(args.candidate_outputs)),
        verifier_choices_payload=load_json(Path(args.verifier_choices)),
        min_choice_count=args.min_choice_count,
        min_confidence_margin=args.min_confidence_margin,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(payload["metadata"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
