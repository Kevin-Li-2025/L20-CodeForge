#!/usr/bin/env python3
"""Cross-check published aggregate receipts; does not replay remote generations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_PACKAGE = Path(__file__).resolve().parents[1] / (
    "benchmarks/code_rlvr_retention_v2_2026_08_30"
)


def equal(actual: Any, expected: Any, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def verify(package: Path) -> dict[str, Any]:
    def read(relative: str) -> dict[str, Any]:
        return json.loads((package / relative).read_text(encoding="utf-8"))

    summary = read("campaign_summary.json")
    selection = read("receipts/comparisons/rlvr-selection.json")
    base = summary["development"]["base"]

    def score(stage: str, split: str, expected_passes: int, tasks: int) -> dict[str, Any]:
        report = read(f"receipts/development/{stage}-{split}.report.json")
        metrics = report.get("summary", report)
        for field in ("tasks", "expected_tasks", "samples"):
            equal(metrics[field], tasks, f"{stage}/{split}/{field}")
        equal(metrics["greedy_passes"], expected_passes, f"{stage}/{split}/passes")
        if not 0 <= expected_passes <= tasks or not math.isclose(
            metrics["greedy_accuracy"], expected_passes / tasks, abs_tol=1e-12
        ):
            raise ValueError(f"{stage}/{split}: inconsistent accuracy")
        return report

    def compare_selection(row: dict[str, Any], report: dict[str, Any], label: str) -> None:
        metrics = report.get("summary", report)
        for field, source in (("tasks", "tasks"), ("passed", "greedy_passes")):
            equal(row[field], metrics[source], f"{label}/{field}")
        equal(row["sha256"], report["output_sha256"], f"{label}/sha256")
        equal(row["greedy_accuracy"], metrics["greedy_accuracy"], f"{label}/accuracy")

    for split, key, selector_key in (
        ("new-dev", "rstar", "base_target"),
        ("mbpp-validation", "mbpp_validation", "base_retention"),
    ):
        report = score("base", split, base[f"{key}_passed"], base[f"{key}_tasks"])
        compare_selection(selection[selector_key], report, selector_key)

    threshold = base["rstar_passed"] + math.ceil(
        summary["protocol"]["required_base_to_rlvr_absolute_delta"] * base["rstar_tasks"] - 1e-12
    )
    equal(selection["required_target_passes"], threshold, "selection threshold")
    equal(
        selection["required_absolute_delta"],
        summary["protocol"]["required_base_to_rlvr_absolute_delta"],
        "selection delta",
    )
    candidates = {row["name"]: row for row in selection["candidates"]}
    equal(len(candidates), len(selection["candidates"]), "unique seeds")
    equal(sorted(candidates), ["seed42", "seed43", "seed44"], "seed coverage")
    eligible = []
    for seed, candidate in candidates.items():
        result = summary["development"][seed]
        for stage in ("sft", "rlvr"):
            target = score(
                f"{stage}-{seed}", "new-dev", result[f"{stage}_rstar_passed"], base["rstar_tasks"]
            )
            retention = score(
                f"{stage}-{seed}",
                "mbpp-validation",
                result[f"{stage}_mbpp_validation_passed"],
                base["mbpp_validation_tasks"],
            )
            if stage == "rlvr":
                compare_selection(candidate["target"], target, f"{seed}/target")
                compare_selection(candidate["retention"], retention, f"{seed}/retention")
        target_gate = result["rlvr_rstar_passed"] >= threshold
        retention_gate = result["rlvr_mbpp_validation_passed"] >= base["mbpp_validation_passed"]
        is_eligible = target_gate and retention_gate
        equal(candidate["target_gate_passed"], target_gate, f"{seed}/target gate")
        equal(candidate["retention_gate_passed"], retention_gate, f"{seed}/retention gate")
        equal(candidate["eligible"], is_eligible, f"{seed}/selection eligible")
        equal(result["eligible"], is_eligible, f"{seed}/summary eligible")
        if is_eligible:
            eligible.append(candidate)
    eligible.sort(
        key=lambda row: (-row["target"]["passed"], -row["retention"]["passed"], row["name"])
    )
    selected = eligible[0] if eligible else None
    equal(selection["selected"], selected, "selected candidate receipt")
    equal(summary["selected_candidate"], selected["name"] if selected else None, "selected seed")
    for report in (selection, summary):
        equal(report["development_gate_passed"], bool(eligible), "development gate")

    guardrail = read("receipts/final/evalplus-no-regression.json")
    passed = []
    for name, report in guardrail["datasets"].items():
        result = summary["report_only_final"]["evalplus"][f"{name}plus"]
        base_passes, final_passes = report["base"]["plus_passed"], report["final"]["plus_passed"]
        for field, value in (
            ("base_passed", base_passes),
            ("selected_passed", final_passes),
            ("tasks", report["expected_tasks"]),
            ("delta_tasks", final_passes - base_passes),
        ):
            equal(result[field], value, f"{name}/{field}")
        no_regression = final_passes >= base_passes
        equal(result["no_regression"], no_regression, f"{name}/summary gate")
        equal(report["no_regression"], no_regression, f"{name}/receipt gate")
        passed.append(no_regression)
    equal(sorted(guardrail["datasets"]), ["humaneval", "mbpp"], "guardrail coverage")
    for report in (summary, guardrail):
        equal(report["evalplus_no_regression_passed"], all(passed), "EvalPlus gate")
    # This verifier is for the frozen failed campaign, not a future result template.
    equal(summary["status"], "FAIL_EVALPLUS_NO_REGRESSION", "campaign status")
    equal(summary["target_achieved"], False, "quality target")
    equal(all(passed), False, "failed guardrail")
    return {
        "receipt_consistency": "PASS",
        "model_quality_target_achieved": False,
        "scope": "Aggregate receipt consistency only; remote raw samples not replayed.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    print(json.dumps(verify(parser.parse_args().package), indent=2))
