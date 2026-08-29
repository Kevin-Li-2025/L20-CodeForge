from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from l20_codeforge.data.io import read_jsonl, write_json
from l20_codeforge.rewards.code_execution import (
    CodeExecutionConfig,
    CodeExecutionReport,
    CodeTestCase,
    evaluate_python_completion,
)


class VerifierCandidate(BaseModel):
    candidate_id: str
    code: str
    expected_correct: bool
    source: str | None = None
    tags: list[str] = Field(default_factory=list)


class VerifierAuditRecord(BaseModel):
    task_id: str
    tests: list[CodeTestCase]
    reference_solutions: list[str]
    candidates: list[VerifierCandidate] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class CandidateAudit(BaseModel):
    candidate_id: str
    expected_correct: bool
    compiled: bool
    accepted: bool
    behavior_signature: str
    reward: float
    tests_passed: int
    tests_total: int
    source: str | None = None
    tags: list[str] = Field(default_factory=list)


class TaskVerifierAudit(BaseModel):
    task_id: str
    admitted: bool
    reference_count: int
    references_accepted: int
    reference_signatures: list[str]
    candidates: list[CandidateAudit]
    faulty_behavior_count: int
    faulty_candidate_count: int
    faulty_compile_failure_count: int
    faulty_candidates_killed: int
    correct_candidate_count: int
    correct_candidates_accepted: int
    test_faulty_kill_counts: list[int]
    notes: list[str] = Field(default_factory=list)


class VerifierAuditGates(BaseModel):
    min_reference_accept_rate: float = 1.0
    min_faulty_kill_rate: float = 0.7
    max_false_positive_rate: float = 0.05
    max_false_negative_rate: float = 0.05


def audit_verifier_dataset(
    input_path: Path,
    output_path: Path,
    execution_config: CodeExecutionConfig | None = None,
    gates: VerifierAuditGates | None = None,
    min_reference_solutions: int = 1,
    limit: int | None = None,
) -> dict[str, object]:
    active_config = execution_config or CodeExecutionConfig()
    active_gates = gates or VerifierAuditGates()
    task_audits: list[TaskVerifierAudit] = []
    for index, record in enumerate(read_jsonl(input_path, VerifierAuditRecord)):
        if limit is not None and index >= limit:
            break
        task_audits.append(
            audit_verifier_record(
                record,
                execution_config=active_config,
                min_reference_solutions=min_reference_solutions,
            )
        )

    summary = _summarize(task_audits)
    gate_results = {
        "reference_accept_rate": {
            "value": summary["reference_accept_rate"],
            "operator": ">=",
            "threshold": active_gates.min_reference_accept_rate,
            "passed": summary["reference_accept_rate"]
            >= active_gates.min_reference_accept_rate,
        },
        "faulty_candidate_kill_rate": {
            "value": summary["faulty_candidate_kill_rate"],
            "operator": ">=",
            "threshold": active_gates.min_faulty_kill_rate,
            "passed": summary["faulty_candidate_count"] > 0
            and summary["faulty_candidate_kill_rate"] >= active_gates.min_faulty_kill_rate,
        },
        "false_positive_rate": {
            "value": summary["false_positive_rate"],
            "operator": "<=",
            "threshold": active_gates.max_false_positive_rate,
            "passed": summary["faulty_candidate_count"] > 0
            and summary["false_positive_rate"] <= active_gates.max_false_positive_rate,
        },
        "false_negative_rate": {
            "value": summary["false_negative_rate"],
            "operator": "<=",
            "threshold": active_gates.max_false_negative_rate,
            "passed": summary["correct_candidate_count"] > 0
            and summary["false_negative_rate"] <= active_gates.max_false_negative_rate,
        },
    }
    status = "PASS" if task_audits and all(item["passed"] for item in gate_results.values()) else "FAIL"
    payload: dict[str, object] = {
        "status": status,
        "source": "failure_driven_code_verifier_audit_v1",
        "input": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "execution_config": active_config.model_dump(),
        "gates": gate_results,
        "summary": summary,
        "tasks": [audit.model_dump() for audit in task_audits],
        "claim_boundary": (
            "A PASS validates this labeled verifier-audit set only. It is not a model-capability "
            "or frozen-benchmark result. The subprocess limits are not a security sandbox."
        ),
    }
    write_json(output_path, payload)
    return payload


def audit_verifier_record(
    record: VerifierAuditRecord,
    execution_config: CodeExecutionConfig | None = None,
    min_reference_solutions: int = 1,
) -> TaskVerifierAudit:
    active_config = execution_config or CodeExecutionConfig()
    reference_reports = [
        evaluate_python_completion(solution, record.tests, active_config)
        for solution in record.reference_solutions
    ]
    candidate_reports: list[tuple[VerifierCandidate, CodeExecutionReport]] = [
        (
            candidate,
            evaluate_python_completion(candidate.code, record.tests, active_config),
        )
        for candidate in record.candidates
    ]

    references_accepted = sum(report.all_passed for report in reference_reports)
    admitted = (
        len(reference_reports) >= min_reference_solutions
        and references_accepted == len(reference_reports)
    )
    notes = []
    if len(reference_reports) < min_reference_solutions:
        notes.append("insufficient_reference_solutions")
    if references_accepted != len(reference_reports):
        notes.append("reference_solution_rejected")

    candidates = [
        CandidateAudit(
            candidate_id=candidate.candidate_id,
            expected_correct=candidate.expected_correct,
            compiled=report.compiled,
            accepted=report.all_passed,
            behavior_signature=report.behavior_signature,
            reward=report.reward,
            tests_passed=report.tests_passed,
            tests_total=report.tests_total,
            source=candidate.source,
            tags=candidate.tags,
        )
        for candidate, report in candidate_reports
    ]
    faulty = [item for item in candidates if not item.expected_correct]
    correct = [item for item in candidates if item.expected_correct]
    test_faulty_kill_counts = [
        sum(
            not report.test_results[test_index].passed
            for candidate, report in candidate_reports
            if not candidate.expected_correct and report.compiled
        )
        for test_index in range(len(record.tests))
    ]
    return TaskVerifierAudit(
        task_id=record.task_id,
        admitted=admitted,
        reference_count=len(reference_reports),
        references_accepted=references_accepted,
        reference_signatures=[report.behavior_signature for report in reference_reports],
        candidates=candidates,
        faulty_behavior_count=len({item.behavior_signature for item in faulty}),
        faulty_candidate_count=len(faulty),
        faulty_compile_failure_count=sum(not item.compiled for item in faulty),
        faulty_candidates_killed=sum(not item.accepted for item in faulty),
        correct_candidate_count=len(correct),
        correct_candidates_accepted=sum(item.accepted for item in correct),
        test_faulty_kill_counts=test_faulty_kill_counts,
        notes=notes,
    )


def _summarize(task_audits: list[TaskVerifierAudit]) -> dict[str, int | float]:
    reference_count = sum(item.reference_count for item in task_audits)
    references_accepted = sum(item.references_accepted for item in task_audits)
    faulty_count = sum(item.faulty_candidate_count for item in task_audits)
    faulty_killed = sum(item.faulty_candidates_killed for item in task_audits)
    correct_count = sum(item.correct_candidate_count for item in task_audits)
    correct_accepted = sum(item.correct_candidates_accepted for item in task_audits)
    return {
        "task_count": len(task_audits),
        "admitted_task_count": sum(item.admitted for item in task_audits),
        "reference_count": reference_count,
        "references_accepted": references_accepted,
        "reference_accept_rate": _safe_ratio(references_accepted, reference_count),
        "faulty_candidate_count": faulty_count,
        "faulty_compile_failure_count": sum(
            item.faulty_compile_failure_count for item in task_audits
        ),
        "faulty_candidates_killed": faulty_killed,
        "faulty_candidate_kill_rate": _safe_ratio(faulty_killed, faulty_count),
        "false_positive_rate": _safe_ratio(faulty_count - faulty_killed, faulty_count),
        "correct_candidate_count": correct_count,
        "correct_candidates_accepted": correct_accepted,
        "correct_candidate_accept_rate": _safe_ratio(correct_accepted, correct_count),
        "false_negative_rate": _safe_ratio(correct_count - correct_accepted, correct_count),
        "faulty_behavior_count": sum(item.faulty_behavior_count for item in task_audits),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
