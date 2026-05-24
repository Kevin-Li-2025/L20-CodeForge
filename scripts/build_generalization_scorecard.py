#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def as_float(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def load_evalplus_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in ("base_pass@1", "plus_pass@1", "base_pass@10", "plus_pass@10"):
                parsed[key] = as_float(parsed.get(key))
            rows.append(parsed)
    return rows


def index_evalplus_rows(rows: list[dict[str, Any]], protocol: str) -> dict[str, dict[str, Any]]:
    return {
        row["dataset"]: row
        for row in rows
        if row.get("protocol") == protocol
    }


def evalplus_generalization(rows: list[dict[str, Any]]) -> dict[str, Any]:
    greedy = index_evalplus_rows(rows, "greedy_model")
    clean = index_evalplus_rows(rows, "clean_public_signal_system")
    comparisons = []
    for dataset in sorted(set(greedy) & set(clean)):
        baseline = greedy[dataset]
        candidate = clean[dataset]
        base_delta = candidate["base_pass@1"] - baseline["base_pass@1"]
        plus_delta = candidate["plus_pass@1"] - baseline["plus_pass@1"]
        comparisons.append(
            {
                "dataset": dataset,
                "baseline": baseline["name"],
                "candidate": candidate["name"],
                "base_pass_at_1": {
                    "baseline": baseline["base_pass@1"],
                    "candidate": candidate["base_pass@1"],
                    "delta": round(base_delta, 6),
                },
                "plus_pass_at_1": {
                    "baseline": baseline["plus_pass@1"],
                    "candidate": candidate["plus_pass@1"],
                    "delta": round(plus_delta, 6),
                },
            }
        )
    return {"comparisons": comparisons}


def metric_value(item: dict[str, Any]) -> float:
    if "score" in item:
        return float(item["score"])
    return float(item["pass_at_1"])


def build_slice_gains(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    gains = []
    for group_name in sorted(set(baseline) | set(candidate)):
        baseline_item = baseline.get(group_name)
        candidate_item = candidate.get(group_name)
        if not baseline_item or not candidate_item:
            continue
        baseline_score = metric_value(baseline_item)
        candidate_score = metric_value(candidate_item)
        gains.append(
            {
                "slice": group_name,
                "baseline_passed": baseline_item["passed"],
                "candidate_passed": candidate_item["passed"],
                "total": candidate_item["total"],
                "baseline_pass_at_1": baseline_score,
                "candidate_pass_at_1": candidate_score,
                "delta": round(candidate_score - baseline_score, 6),
            }
        )
    return gains


def lcb_generalization(summary: dict[str, Any]) -> dict[str, Any]:
    score = summary["score"]
    baseline = summary["baseline_greedy"]
    breakdown = summary["breakdown"]
    baseline_breakdown = summary["baseline_breakdown"]
    return {
        "overall": {
            "baseline_passed": baseline["passed"],
            "candidate_passed": score["passed"],
            "total": score["total"],
            "baseline_pass_at_1": baseline["pass_at_1"],
            "candidate_pass_at_1": score["pass_at_1"],
            "delta": round(score["pass_at_1"] - baseline["pass_at_1"], 6),
            "relative_gain": round(
                (score["pass_at_1"] - baseline["pass_at_1"]) / baseline["pass_at_1"],
                6,
            ),
        },
        "difficulty": build_slice_gains(
            baseline_breakdown["difficulty"],
            breakdown["difficulty"],
        ),
        "platform": build_slice_gains(
            baseline_breakdown["platform"],
            breakdown["platform"],
        ),
    }


def evaluate_gate(
    lcb: dict[str, Any],
    evalplus: dict[str, Any],
    min_lcb_delta: float,
    min_slice_delta: float,
    max_evalplus_plus_regression: float,
) -> dict[str, Any]:
    checks = []
    checks.append(
        {
            "name": "lcb_overall_improves",
            "value": lcb["overall"]["delta"],
            "threshold": min_lcb_delta,
            "passed": lcb["overall"]["delta"] >= min_lcb_delta,
        }
    )
    for split_name in ("difficulty", "platform"):
        for item in lcb[split_name]:
            checks.append(
                {
                    "name": f"lcb_{split_name}_{item['slice']}_not_regressed",
                    "value": item["delta"],
                    "threshold": min_slice_delta,
                    "passed": item["delta"] >= min_slice_delta,
                }
            )
    for item in evalplus["comparisons"]:
        checks.append(
            {
                "name": f"evalplus_{item['dataset']}_plus_not_regressed",
                "value": item["plus_pass_at_1"]["delta"],
                "threshold": -max_evalplus_plus_regression,
                "passed": item["plus_pass_at_1"]["delta"] >= -max_evalplus_plus_regression,
            }
        )
    return {
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "checks": checks,
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_readme(scorecard: dict[str, Any]) -> str:
    lcb = scorecard["livecodebench"]
    evalplus = scorecard["evalplus"]
    gate = scorecard["gate"]
    lcb_rows = [
        [
            "overall",
            lcb["overall"]["baseline_passed"],
            lcb["overall"]["candidate_passed"],
            lcb["overall"]["total"],
            f"{lcb['overall']['baseline_pass_at_1']:.4f}",
            f"{lcb['overall']['candidate_pass_at_1']:.4f}",
            f"{lcb['overall']['delta']:+.4f}",
        ]
    ]
    for split_name in ("difficulty", "platform"):
        for item in lcb[split_name]:
            lcb_rows.append(
                [
                    f"{split_name}:{item['slice']}",
                    item["baseline_passed"],
                    item["candidate_passed"],
                    item["total"],
                    f"{item['baseline_pass_at_1']:.4f}",
                    f"{item['candidate_pass_at_1']:.4f}",
                    f"{item['delta']:+.4f}",
                ]
            )
    evalplus_rows = [
        [
            item["dataset"],
            item["baseline"],
            item["candidate"],
            f"{item['plus_pass_at_1']['baseline']:.3f}",
            f"{item['plus_pass_at_1']['candidate']:.3f}",
            f"{item['plus_pass_at_1']['delta']:+.3f}",
        ]
        for item in evalplus["comparisons"]
    ]
    failed_checks = [check for check in gate["checks"] if not check["passed"]]
    return (
        "# Generalization Scorecard\n\n"
        "This scorecard is a guardrail against LiveCodeBench-only overfitting. "
        "Hidden/private tests are used only for final measurement; selector, "
        "repair, and training changes should continue to pass this cross-benchmark gate.\n\n"
        f"Gate status: `{gate['status']}`\n\n"
        "## LiveCodeBench Full `release_v6`\n\n"
        + markdown_table(
            ["slice", "greedy passed", "candidate passed", "total", "greedy", "candidate", "delta"],
            lcb_rows,
        )
        + "\n\n"
        "## EvalPlus Holdout\n\n"
        + markdown_table(
            ["dataset", "greedy", "clean system", "greedy plus", "system plus", "delta"],
            evalplus_rows,
        )
        + "\n\n"
        "## Gate Checks\n\n"
        + (
            "All checks passed.\n"
            if not failed_checks
            else "\n".join(
                f"- `{check['name']}` value `{check['value']}` below threshold `{check['threshold']}`"
                for check in failed_checks
            )
            + "\n"
        )
    )


def build_scorecard(args: argparse.Namespace) -> dict[str, Any]:
    lcb_summary = json.loads(Path(args.lcb_summary).read_text(encoding="utf-8"))
    evalplus_rows = load_evalplus_rows(Path(args.evalplus_summary))
    lcb = lcb_generalization(lcb_summary)
    evalplus = evalplus_generalization(evalplus_rows)
    gate = evaluate_gate(
        lcb,
        evalplus,
        min_lcb_delta=args.min_lcb_delta,
        min_slice_delta=args.min_slice_delta,
        max_evalplus_plus_regression=args.max_evalplus_plus_regression,
    )
    return {
        "schema_version": 1,
        "purpose": "cross-benchmark generalization guardrail",
        "livecodebench": lcb,
        "evalplus": evalplus,
        "gate": gate,
        "inputs": {
            "lcb_summary": args.lcb_summary,
            "evalplus_summary": args.evalplus_summary,
        },
        "thresholds": {
            "min_lcb_delta": args.min_lcb_delta,
            "min_slice_delta": args.min_slice_delta,
            "max_evalplus_plus_regression": args.max_evalplus_plus_regression,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cross-benchmark generalization scorecard.")
    parser.add_argument(
        "--lcb-summary",
        default=(
            "benchmarks/livecodebench_full_release_v6_2026_05_22/"
            "full_n8_public_select_summary.json"
        ),
    )
    parser.add_argument(
        "--evalplus-summary",
        default="benchmarks/evalplus_l20_codeforge_2026_05_22/summary.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-lcb-delta", type=float, default=0.0)
    parser.add_argument("--min-slice-delta", type=float, default=-0.005)
    parser.add_argument("--max-evalplus-plus-regression", type=float, default=0.01)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard = build_scorecard(args)
    (output_dir / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(render_readme(scorecard), encoding="utf-8")
    print(json.dumps(scorecard["gate"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
