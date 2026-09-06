from __future__ import annotations

import json
from pathlib import Path

from l20_codeforge.evals.verifier_audit import (
    VerifierAuditGates,
    VerifierAuditRecord,
    audit_verifier_dataset,
    audit_verifier_record,
)


def _record() -> VerifierAuditRecord:
    return VerifierAuditRecord.model_validate(
        {
            "task_id": "square",
            "tests": [
                {"name": "positive", "input": "2\n", "output": "4\n"},
                {"name": "negative", "input": "-3\n", "output": "9\n"},
            ],
            "reference_solutions": [
                "x = int(input())\nprint(x * x)\n",
                "x = int(input())\nprint(pow(x, 2))\n",
            ],
            "candidates": [
                {
                    "candidate_id": "known-correct",
                    "code": "x = int(input())\nprint(abs(x) ** 2)\n",
                    "expected_correct": True,
                },
                {
                    "candidate_id": "near-correct",
                    "code": "x = int(input())\nprint(x + 2)\n",
                    "expected_correct": False,
                    "tags": ["boundary-sign"],
                },
                {
                    "candidate_id": "constant",
                    "code": "print(9)\n",
                    "expected_correct": False,
                },
                {
                    "candidate_id": "syntax-error",
                    "code": "def broken(:\n",
                    "expected_correct": False,
                },
            ],
        }
    )


def test_audit_verifier_record_tracks_kill_rate_and_behavior_diversity() -> None:
    audit = audit_verifier_record(_record(), min_reference_solutions=2)

    assert audit.admitted is True
    assert audit.references_accepted == 2
    assert audit.faulty_candidate_count == 3
    assert audit.faulty_compile_failure_count == 1
    assert audit.faulty_candidates_killed == 3
    assert audit.correct_candidates_accepted == 1
    assert audit.faulty_behavior_count == 3
    assert audit.test_faulty_kill_counts == [1, 1]


def test_audit_verifier_dataset_writes_hash_and_passes_gates(tmp_path: Path) -> None:
    input_path = tmp_path / "verifier.jsonl"
    output_path = tmp_path / "audit.json"
    input_path.write_text(_record().model_dump_json() + "\n", encoding="utf-8")

    report = audit_verifier_dataset(
        input_path=input_path,
        output_path=output_path,
        gates=VerifierAuditGates(
            min_reference_accept_rate=1.0,
            min_faulty_kill_rate=1.0,
            max_false_positive_rate=0.0,
            max_false_negative_rate=0.0,
        ),
        min_reference_solutions=2,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert saved["status"] == "PASS"
    assert len(saved["input_sha256"]) == 64
    assert saved["summary"]["faulty_candidate_kill_rate"] == 1.0
    assert "not a model-capability" in saved["claim_boundary"]


def test_audit_verifier_dataset_fails_when_faulty_code_passes(tmp_path: Path) -> None:
    record = _record()
    record.tests = record.tests[:1]
    input_path = tmp_path / "weak.jsonl"
    input_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

    report = audit_verifier_dataset(
        input_path=input_path,
        output_path=tmp_path / "weak-audit.json",
        gates=VerifierAuditGates(
            min_reference_accept_rate=1.0,
            min_faulty_kill_rate=0.7,
            max_false_positive_rate=0.05,
            max_false_negative_rate=0.0,
        ),
        min_reference_solutions=2,
    )

    assert report["status"] == "FAIL"
    assert report["summary"]["false_positive_rate"] > 0.0
    assert report["gates"]["false_positive_rate"]["passed"] is False
