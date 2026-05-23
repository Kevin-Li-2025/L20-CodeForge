from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


def load_audit_module():
    script = Path(__file__).parents[1] / "scripts" / "audit_lcb_expected_verifier.py"
    spec = importlib.util.spec_from_file_location("audit_lcb_expected_verifier", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prompt_id(question_id: str, input_index: int) -> str:
    digest = hashlib.sha256(f"{question_id}:{input_index}".encode()).hexdigest()[:12]
    return f"{question_id}:{input_index}:{digest}"


def test_expected_verifier_audit_sweeps_stricter_thresholds() -> None:
    audit_mod = load_audit_module()
    public_selection_payload = {
        "records": [
            {
                "question_id": "q1",
                "question_title": "Q1",
                "selected_index": 0,
                "public_scores": [1.0, 1.0],
            }
        ]
    }
    candidate_outputs_payload = {
        "candidate_outputs": [
            {
                "question_id": "q1",
                "input_index": 0,
                "input_sha256_16": "abc",
                "options": [
                    {"label": "A", "candidate_indices": [0]},
                    {"label": "B", "candidate_indices": [1]},
                ],
            }
        ]
    }
    verifier_choices_payload = {
        "records": [
            {
                "record_id": prompt_id("q1", 0),
                "question_id": "q1",
                "choice": "B",
                "confidence": 1.0,
            },
            {
                "record_id": prompt_id("q1", 0),
                "question_id": "q1",
                "choice": "B",
                "confidence": 1.0,
            },
        ]
    }
    current_candidate_selection = {
        "q1": {
            "question_id": "q1",
            "selected_index": 1,
            "verifier_selected_index": 1,
            "verifier_selected_choice_count": 2,
            "verifier_confidence_margin_vs_public": 2.0,
            "expected_verifier_scores": {
                "0": {"confidence_sum": 0.0, "choice_count": 0},
                "1": {"confidence_sum": 2.0, "choice_count": 2},
            },
        }
    }
    comparison_summary = {
        "total": 1,
        "baseline_passed": 0,
        "candidate_passed": 1,
        "changed_or_outcome_different": [
            {
                "question_id": "q1",
                "question_title": "Q1",
                "baseline_selected_index": 0,
                "candidate_selected_index": 1,
                "baseline_pass": False,
                "candidate_pass": True,
                "stabilized_candidate_pass": True,
                "selection_changed": True,
            }
        ],
    }

    audit = audit_mod.build_audit(
        comparison_summary=comparison_summary,
        public_selection_payload=public_selection_payload,
        candidate_outputs_payload=candidate_outputs_payload,
        verifier_choices_payload=verifier_choices_payload,
        baseline_eval={"q1": {"passed": False}},
        candidate_eval={"q1": {"passed": True}},
        baseline_selection={
            "q1": {
                "question_id": "q1",
                "selected_index": 0,
                "public_scores": [1.0, 1.0],
            }
        },
        current_candidate_selection=current_candidate_selection,
        min_choice_counts=[2, 3],
        min_confidence_margins=[1.0],
    )

    assert audit["override_audit_records"][0]["choice_votes"]["B"]["choice_count"] == 2
    assert audit["threshold_sweep"][0]["target_override_question_ids"] == ["q1"]
    assert audit["threshold_sweep"][0]["inferred_raw_net_gain"] == 1
    assert audit["threshold_sweep"][1]["target_override_question_ids"] == []
    assert audit["threshold_sweep"][1]["inferred_raw_net_gain"] == 0
