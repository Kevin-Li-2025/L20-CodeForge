#!/usr/bin/env python3
"""Compare LiveCodeBench selectors on one exactly shared candidate pool.

The hidden all-candidate evaluation is used only to score already-defined
selection rules. It is never used to choose a candidate, except for the
explicitly labelled oracle ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_unique(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = str(record["question_id"])
        if question_id in indexed:
            raise ValueError(f"duplicate question_id in {label}: {question_id}")
        indexed[question_id] = record
    return indexed


def wilson_interval(passed: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("total must be positive")
    rate = passed / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half_width = (
        z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    )
    return [center - half_width, center + half_width]


def exact_paired_binomial_pvalue(left_wins: int, right_wins: int) -> float:
    """Two-sided exact sign-test p-value after excluding paired ties."""
    discordant = left_wins + right_wins
    if discordant == 0:
        return 1.0
    tail = min(left_wins, right_wins)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def deterministic_random_index(seed: int, question_id: str, candidate_count: int) -> int:
    payload = f"{seed}\0{question_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % candidate_count


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    return ordered[round(probability * (len(ordered) - 1))]


def selector_summary(
    evaluations: list[dict[str, Any]],
    selected_indices: dict[str, int],
    oracle_available_ids: set[str] | None = None,
) -> dict[str, Any]:
    outcomes: dict[str, bool] = {}
    by_difficulty: dict[str, list[bool]] = {}
    for record in evaluations:
        question_id = str(record["question_id"])
        selected_index = selected_indices[question_id]
        grades = record["graded_list"]
        if not 0 <= selected_index < len(grades):
            raise ValueError(f"selected index out of range for {question_id}: {selected_index}")
        passed = bool(grades[selected_index])
        outcomes[question_id] = passed
        by_difficulty.setdefault(str(record.get("difficulty", "unknown")), []).append(passed)

    passed_count = sum(outcomes.values())
    total = len(outcomes)
    summary = {
        "passed": passed_count,
        "total": total,
        "pass_rate": passed_count / total,
        "wilson_95_ci": wilson_interval(passed_count, total),
        "by_difficulty": {
            difficulty: {
                "passed": sum(group),
                "total": len(group),
                "pass_rate": sum(group) / len(group),
            }
            for difficulty, group in sorted(by_difficulty.items())
        },
        "outcomes": outcomes,
    }
    if oracle_available_ids is not None:
        selected_when_available = sum(outcomes[question_id] for question_id in oracle_available_ids)
        summary["oracle_available_tasks"] = len(oracle_available_ids)
        summary["selector_success_given_oracle_available"] = (
            selected_when_available / len(oracle_available_ids) if oracle_available_ids else None
        )
        summary["missed_available_solutions"] = len(oracle_available_ids) - selected_when_available
    return summary


def paired_comparison(
    left_name: str,
    left: dict[str, bool],
    right_name: str,
    right: dict[str, bool],
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("paired selector outcomes must cover the same question IDs")
    left_wins = sum(left[qid] and not right[qid] for qid in left)
    right_wins = sum(right[qid] and not left[qid] for qid in left)
    return {
        "left": left_name,
        "right": right_name,
        "left_only_passes": left_wins,
        "right_only_passes": right_wins,
        "paired_ties": len(left) - left_wins - right_wins,
        "two_sided_exact_sign_test_p": exact_paired_binomial_pvalue(left_wins, right_wins),
    }


def build_comparison(
    generations: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    public_records: list[dict[str, Any]],
    behavior_records: list[dict[str, Any]],
    *,
    random_trials: int = 10_000,
    random_seed_to_report: int = 42,
) -> dict[str, Any]:
    generation_by_id = index_unique(generations, "generations")
    if not generation_by_id:
        raise ValueError("candidate pool must not be empty")
    evaluation_by_id = index_unique(evaluations, "hidden evaluation")
    public_by_id = index_unique(public_records, "public selection")
    behavior_by_id = index_unique(behavior_records, "behavior selection")
    id_sets = [
        set(mapping)
        for mapping in (generation_by_id, evaluation_by_id, public_by_id, behavior_by_id)
    ]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError("generation, evaluation, public, and behavior IDs do not match exactly")
    if type(random_trials) is not int or random_trials <= 0:
        raise ValueError("random_trials must be positive")

    candidate_counts: set[int] = set()
    ordered_evaluations: list[dict[str, Any]] = []
    for question_id in sorted(evaluation_by_id):
        generation = generation_by_id[question_id]
        evaluation = evaluation_by_id[question_id]
        codes = generation["code_list"]
        if evaluation["code_list"] != codes:
            raise ValueError(f"candidate code/order mismatch for {question_id}")
        grades = evaluation["graded_list"]
        if len(grades) != len(codes) or not codes:
            raise ValueError(f"candidate/grade length mismatch for {question_id}")
        if any(type(grade) is not bool for grade in grades):
            raise ValueError(f"hidden grades must be JSON booleans for {question_id}")
        candidate_counts.add(len(codes))
        for label, selection in (
            ("public", public_by_id[question_id]),
            ("behavior", behavior_by_id[question_id]),
        ):
            if type(selection["n_candidates"]) is not int or selection["n_candidates"] != len(
                codes
            ):
                raise ValueError(f"{label} candidate count mismatch for {question_id}")
            index = selection["selected_index"]
            if type(index) is not int or not 0 <= index < len(codes):
                raise ValueError(
                    f"{label} selected index out of range or not integer for {question_id}"
                )
        ordered_evaluations.append(evaluation)
    if len(candidate_counts) != 1:
        raise ValueError(f"candidate budget is not constant: {sorted(candidate_counts)}")
    candidate_count = candidate_counts.pop()

    selectors = {
        "first_candidate": {qid: 0 for qid in evaluation_by_id},
        "public_tests": {
            qid: int(record["selected_index"]) for qid, record in public_by_id.items()
        },
        "behavior_consensus": {
            qid: int(record["selected_index"]) for qid, record in behavior_by_id.items()
        },
    }
    oracle_available_ids = {
        str(record["question_id"]) for record in ordered_evaluations if any(record["graded_list"])
    }
    fixed = {
        name: selector_summary(ordered_evaluations, indices, oracle_available_ids)
        for name, indices in selectors.items()
    }

    random_rates: list[float] = []
    reported_random: dict[str, Any] | None = None
    for seed in range(random_trials):
        indices = {
            qid: deterministic_random_index(seed, qid, candidate_count) for qid in evaluation_by_id
        }
        summary = selector_summary(ordered_evaluations, indices)
        random_rates.append(summary["pass_rate"])
        if seed == random_seed_to_report:
            reported_random = summary
    if reported_random is None:
        indices = {
            qid: deterministic_random_index(random_seed_to_report, qid, candidate_count)
            for qid in evaluation_by_id
        }
        reported_random = selector_summary(ordered_evaluations, indices)

    random_expected = math.fsum(
        sum(bool(grade) for grade in record["graded_list"]) / len(record["graded_list"])
        for record in ordered_evaluations
    ) / len(ordered_evaluations)
    oracle_passed = sum(any(record["graded_list"]) for record in ordered_evaluations)

    comparisons = [
        paired_comparison(
            "public_tests",
            fixed["public_tests"]["outcomes"],
            "first_candidate",
            fixed["first_candidate"]["outcomes"],
        ),
        paired_comparison(
            "behavior_consensus",
            fixed["behavior_consensus"]["outcomes"],
            "public_tests",
            fixed["public_tests"]["outcomes"],
        ),
    ]
    agreement = sum(
        selectors["public_tests"][qid] == selectors["behavior_consensus"][qid]
        for qid in evaluation_by_id
    )
    for summary in fixed.values():
        del summary["outcomes"]
    del reported_random["outcomes"]

    return {
        "status": "complete_posthoc_equal_candidate_budget_comparison",
        "claim_boundary": (
            "Hidden outcomes score frozen selector decisions only. The oracle is a non-deployable "
            "ceiling and no selector is tuned from hidden outcomes in this report."
        ),
        "budget": {
            "tasks": len(ordered_evaluations),
            "candidates_per_task": candidate_count,
            "total_candidates": len(ordered_evaluations) * candidate_count,
            "candidate_pool_identical_for_rescoring": True,
            "selection_source_hashes_verified": False,
        },
        "fixed_selectors": fixed,
        "uniform_random_selector": {
            "analytical_expected_pass_rate": random_expected,
            "reported_seed": random_seed_to_report,
            "reported_seed_result": reported_random,
            "trials": random_trials,
            "trial_mean_pass_rate": math.fsum(random_rates) / len(random_rates),
            "trial_pass_rate_95_interval": [
                quantile(random_rates, 0.025),
                quantile(random_rates, 0.975),
            ],
            "trial_min_pass_rate": min(random_rates),
            "trial_max_pass_rate": max(random_rates),
        },
        "hidden_oracle_ceiling": {
            "passed": oracle_passed,
            "total": len(ordered_evaluations),
            "pass_rate": oracle_passed / len(ordered_evaluations),
            "deployable_selector": False,
            "definition": "At least one candidate in the frozen pool passes hidden tests.",
        },
        "paired_comparisons": comparisons,
        "public_behavior_selection_agreement": {
            "same_index": agreement,
            "total": len(ordered_evaluations),
            "rate": agreement / len(ordered_evaluations),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    fixed = result["fixed_selectors"]
    random_result = result["uniform_random_selector"]
    budget = result["budget"]
    rows = []
    for name in ("first_candidate", "public_tests", "behavior_consensus"):
        summary = fixed[name]
        low, high = summary["wilson_95_ci"]
        success = summary["selector_success_given_oracle_available"]
        success_text = "n/a" if success is None else f"{success:.4f}"
        rows.append(
            f"| `{name}` | {summary['passed']}/{summary['total']} | "
            f"{summary['pass_rate']:.4f} | "
            f"{success_text} | "
            f"[{low:.4f}, {high:.4f}] |"
        )
    seed_result = random_result["reported_seed_result"]
    low, high = seed_result["wilson_95_ci"]
    rows.append(
        f"| `uniform_random_seed_{random_result['reported_seed']}` | "
        f"{seed_result['passed']}/{seed_result['total']} | {seed_result['pass_rate']:.4f} | n/a | "
        f"[{low:.4f}, {high:.4f}] |"
    )
    oracle = result["hidden_oracle_ceiling"]
    comparisons = result["paired_comparisons"]
    return "\n".join(
        [
            "# Equal-budget LiveCodeBench selector comparison",
            "",
            "This is a post-hoc comparison over one frozen candidate pool. Hidden tests are used only",
            "to score frozen choices; the hidden oracle is reported only as a non-deployable ceiling.",
            "",
            "## Budget and provenance boundary",
            "",
            f"- Tasks: `{budget['tasks']}` stratified release-v6 tasks.",
            (
                f"- Candidate budget: exactly `{budget['candidates_per_task']}` candidates per task "
                f"(`{budget['total_candidates']}` total)."
            ),
            (
                "- Every selector is rescored on the same saved candidate strings and order. "
                "Generation is shared, not rerun; this is not a measured equal-total-compute trial."
            ),
            f"- Selector report generation hashes verified: `{budget['selection_source_hashes_verified']}`.",
            "- Selection overhead is not equal: public tests and behavior tests add CPU sandbox work.",
            "- Recorded selection wall times: "
            + "; ".join(
                f"{name} = {fixed[name].get('selection_seconds', 'unavailable')} s"
                for name in ("public_tests", "behavior_consensus")
            )
            + ". These are historical elapsed times, not normalized CPU/GPU costs.",
            "",
            "## Results",
            "",
            "| Selector | Passed | pass@1 | Success given available solution | Wilson 95% CI |",
            "| --- | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            (
                "Uniform-random analytical expectation: "
                f"`{random_result['analytical_expected_pass_rate']:.4f}`. "
                f"Across `{random_result['trials']}` deterministic seeds, its 95% trial range was "
                f"`[{random_result['trial_pass_rate_95_interval'][0]:.4f}, "
                f"{random_result['trial_pass_rate_95_interval'][1]:.4f}]`."
            ),
            "",
            (
                f"Hidden oracle ceiling: `{oracle['passed']}/{oracle['total']}` "
                f"(`{oracle['pass_rate']:.4f}`); this is not a deployable selector."
            ),
            "",
            "## Paired evidence",
            "",
            (
                f"- Public vs first: `{comparisons[0]['left_only_passes']}` public-only passes and "
                f"`{comparisons[0]['right_only_passes']}` first-only passes; exact paired "
                f"`p={comparisons[0]['two_sided_exact_sign_test_p']:.4f}`."
            ),
            (
                f"- Behavior vs public: `{comparisons[1]['left_only_passes']}` behavior-only "
                f"passes and `{comparisons[1]['right_only_passes']}` public-only passes; exact "
                f"paired `p={comparisons[1]['two_sided_exact_sign_test_p']:.4f}`."
            ),
            "",
            "This retrospective candidate-pool comparison is not a fresh held-out benchmark.",
            "The random-seed trial interval describes selector randomness, not uncertainty over new tasks.",
            "Paired tests above must be read alongside effect size, sample size and selection overhead.",
            "",
        ]
    )


def verify_selection_source(report: dict[str, Any], expected_hash: str, label: str) -> None:
    if report.get("generations_sha256") != expected_hash:
        raise ValueError(f"{label} selector generation hash does not match the candidate pool")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--hidden-eval", type=Path, required=True)
    parser.add_argument("--public-selection", type=Path, required=True)
    parser.add_argument("--behavior-selection", type=Path, required=True)
    parser.add_argument("--generation-report", type=Path)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--behavior-report", type=Path, required=True)
    parser.add_argument("--random-trials", type=int, default=10_000)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_sha256 = sha256_file(args.generations)
    for label, path in (("public", args.public_report), ("behavior", args.behavior_report)):
        verify_selection_source(load_json(path), generation_sha256, label)
    input_paths = {
        "generations": args.generations,
        "hidden_eval": args.hidden_eval,
        "public_selection": args.public_selection,
        "behavior_selection": args.behavior_selection,
    }
    result = build_comparison(
        load_json(args.generations),
        load_json(args.hidden_eval),
        load_json(args.public_selection)["records"],
        load_json(args.behavior_selection)["records"],
        random_trials=args.random_trials,
    )
    result["inputs"] = {
        name: {"path": str(path), "sha256": sha256_file(path)} for name, path in input_paths.items()
    }
    result["budget"]["selection_source_hashes_verified"] = True

    if args.generation_report:
        generation_report = load_json(args.generation_report)
        result["budget"]["shared_generation_seconds"] = generation_report.get("generation_seconds")
        result["inputs"]["generation_report"] = {
            "path": str(args.generation_report),
            "sha256": sha256_file(args.generation_report),
        }
    for label, path in (
        ("public_tests", args.public_report),
        ("behavior_consensus", args.behavior_report),
    ):
        if path:
            report = load_json(path)
            selection_seconds = report.get("public_selection_seconds")
            result["fixed_selectors"][label]["selection_seconds"] = selection_seconds
            if isinstance(selection_seconds, (int, float)):
                result["fixed_selectors"][label]["selection_seconds_per_task"] = (
                    selection_seconds / result["budget"]["tasks"]
                )
                generation_seconds = result["budget"].get("shared_generation_seconds")
                if isinstance(generation_seconds, (int, float)) and generation_seconds > 0:
                    result["fixed_selectors"][label]["selection_overhead_vs_generation"] = (
                        selection_seconds / generation_seconds
                    )
            result["inputs"][f"{label}_report"] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
