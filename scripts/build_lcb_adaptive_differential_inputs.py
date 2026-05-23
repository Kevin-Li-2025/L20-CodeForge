#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_lcb_generations import (  # noqa: E402
    bounded_unique,
    differential_behavior_test_indices,
    load_full_jsonl,
    load_generations,
    load_public_selection_payload,
    mutate_functional_input,
    mutate_stdin_input,
    run_behavior_candidate,
    sha256_file,
)


def coerce_public_scores(record: dict[str, Any], n_candidates: int) -> list[float]:
    scores = [float(score) for score in record.get("public_scores", [])[:n_candidates]]
    while len(scores) < n_candidates:
        scores.append(0.0)
    return scores


def public_pass_indices(public_scores: list[float]) -> list[int]:
    return [index for index, score in enumerate(public_scores) if score == 1.0]


def public_inputs(problem: Any) -> list[str]:
    return bounded_unique(
        [
            str(getattr(test_case, "input", ""))
            for test_case in getattr(problem, "public_test_cases", [])
            if str(getattr(test_case, "input", "")).strip()
        ],
        limit=32,
    )


def replace_number_tokens(input_text: str, replacements: dict[int, int]) -> str:
    matches = list(re.finditer(r"-?\d+", input_text))
    pieces: list[str] = []
    cursor = 0
    for token_index, match in enumerate(matches):
        pieces.append(input_text[cursor : match.start()])
        if token_index in replacements:
            pieces.append(str(replacements[token_index]))
        else:
            pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(input_text[cursor:])
    return "".join(pieces)


def adaptive_stdin_mutations(
    input_text: str,
    rng: random.Random,
    limit: int,
    max_input_chars: int,
) -> list[str]:
    variants = list(mutate_stdin_input(input_text, limit=max(1, min(limit // 3, 8))))
    matches = list(re.finditer(r"-?\d+", input_text))
    numbers = [int(match.group(0)) for match in matches]
    if not numbers:
        return bounded_unique(variants, limit)

    edge_values = [-1, 0, 1, 2, 3, 10, 10**9]
    token_count = len(numbers)
    interesting_positions = bounded_unique(
        list(range(min(token_count, 8)))
        + list(range(max(0, token_count - 8), token_count))
        + [rng.randrange(token_count) for _ in range(min(token_count, 8))],
        limit=min(token_count, 24),
    )
    for token_index in interesting_positions:
        original = numbers[token_index]
        replacements = bounded_unique(
            edge_values
            + [
                original - 2,
                original - 1,
                original + 1,
                original + 2,
                original * 2,
                max(0, original - 1),
            ],
            limit=12,
        )
        for replacement in replacements:
            if replacement == original:
                continue
            variants.append(replace_number_tokens(input_text, {token_index: replacement}))
            if len(variants) >= limit * 2:
                break

    if token_count >= 2:
        transforms: list[dict[int, int]] = []
        transforms.append({index: 0 for index in range(token_count)})
        transforms.append({index: 1 for index in range(token_count)})
        transforms.append({index: numbers[index] + 1 for index in range(token_count)})
        transforms.append({index: max(0, numbers[index] - 1) for index in range(token_count)})
        for _ in range(min(32, limit)):
            width = rng.randint(2, min(5, token_count))
            positions = rng.sample(range(token_count), width)
            transforms.append(
                {
                    position: rng.choice(
                        [
                            -1,
                            0,
                            1,
                            2,
                            numbers[position] - 1,
                            numbers[position] + 1,
                            numbers[position] * 2,
                        ]
                    )
                    for position in positions
                }
            )
        for replacements in transforms:
            variants.append(replace_number_tokens(input_text, replacements))

    return bounded_unique(
        [
            variant
            for variant in variants
            if variant != input_text and 0 < len(variant) <= max_input_chars
        ],
        limit,
    )


def mutate_json_random(value: Any, rng: random.Random) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int) and not isinstance(value, bool):
        return rng.choice([-1, 0, 1, 2, value - 1, value + 1, value * 2])
    if isinstance(value, float):
        return rng.choice([-1.0, 0.0, 1.0, value - 1.0, value + 1.0, value * 2.0])
    if isinstance(value, str):
        return rng.choice(["", value[::-1], value + value[:1], value.upper(), value.lower()])
    if isinstance(value, list):
        if not value:
            return [0]
        choice = rng.choice(["empty", "prefix", "reverse", "duplicate", "item"])
        if choice == "empty":
            return []
        if choice == "prefix":
            return value[: rng.randint(1, len(value))]
        if choice == "reverse":
            return list(reversed(value))
        if choice == "duplicate":
            return value + value[-1:]
        changed = list(value)
        index = rng.randrange(len(changed))
        changed[index] = mutate_json_random(changed[index], rng)
        return changed
    if isinstance(value, dict):
        if not value:
            return {"x": 0}
        changed = dict(value)
        key = rng.choice(list(changed))
        changed[key] = mutate_json_random(changed[key], rng)
        return changed
    return value


def adaptive_functional_mutations(
    input_text: str,
    rng: random.Random,
    limit: int,
    max_input_chars: int,
) -> list[str]:
    variants = list(mutate_functional_input(input_text, limit=max(1, min(limit // 3, 8))))
    try:
        values = [json.loads(line) for line in input_text.splitlines() if line.strip()]
    except json.JSONDecodeError:
        return bounded_unique(variants, limit)
    for _ in range(limit * 2):
        if not values:
            break
        changed = list(values)
        index = rng.randrange(len(changed))
        changed[index] = mutate_json_random(changed[index], rng)
        encoded = "\n".join(json.dumps(item, ensure_ascii=True) for item in changed)
        if 0 < len(encoded) <= max_input_chars:
            variants.append(encoded)
    return bounded_unique([variant for variant in variants if variant != input_text], limit)


def build_candidate_inputs(
    problem: Any,
    rng: random.Random,
    max_candidate_inputs: int,
    max_input_chars: int,
) -> list[str]:
    seeds = public_inputs(problem)
    generated: list[str] = []
    is_functional = bool(getattr(problem, "metadata", {}).get("func_name"))
    per_seed_limit = max(8, max_candidate_inputs // max(1, len(seeds)))
    for seed in seeds:
        if is_functional:
            generated.extend(
                adaptive_functional_mutations(
                    seed,
                    rng=rng,
                    limit=per_seed_limit,
                    max_input_chars=max_input_chars,
                )
            )
        else:
            generated.extend(
                adaptive_stdin_mutations(
                    seed,
                    rng=rng,
                    limit=per_seed_limit,
                    max_input_chars=max_input_chars,
                )
            )
    return bounded_unique(generated, max_candidate_inputs)


def choose_differential_inputs(
    behavior_inputs: list[str],
    behavior_outputs: list[list[str]],
    candidate_indices: list[int],
    max_output_inputs: int,
) -> tuple[list[str], list[int]]:
    differential_indices = differential_behavior_test_indices(
        behavior_outputs=behavior_outputs,
        candidate_indices=candidate_indices,
    )
    ranked_indices = sorted(
        differential_indices,
        key=lambda input_index: (
            -len({behavior_outputs[index][input_index] for index in candidate_indices}),
            input_index,
        ),
    )
    selected_indices = ranked_indices[:max_output_inputs]
    return [behavior_inputs[index] for index in selected_indices], selected_indices


def build_adaptive_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.monotonic()
    rng = random.Random(args.seed)
    generation_records = load_generations(Path(args.generations))
    public_selection_payload = load_public_selection_payload(Path(args.public_selection))
    selection_by_id = {
        record["question_id"]: record for record in public_selection_payload["records"]
    }
    question_ids = set(generation_records)
    problems = load_full_jsonl(Path(args.full_jsonl), Path(args.lcb_repo), question_ids)
    if args.limit is not None:
        problems = problems[: args.limit]

    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for problem_number, problem in enumerate(problems, start=1):
        generation = generation_records.get(problem.question_id)
        selection_record = selection_by_id.get(problem.question_id)
        if generation is None or selection_record is None:
            continue
        code_outputs = list(generation.get("code_list") or [])
        if args.max_samples is not None:
            code_outputs = code_outputs[: args.max_samples]
        if not code_outputs:
            continue
        scores = coerce_public_scores(selection_record, len(code_outputs))
        pass_indices = public_pass_indices(scores)
        print(
            f"[{problem_number}/{len(problems)}] {problem.question_id} "
            f"public_pass={pass_indices}",
            flush=True,
        )
        if len(pass_indices) < 2:
            diagnostics.append(
                {
                    "question_id": problem.question_id,
                    "status": "skipped_less_than_two_public_pass",
                    "public_pass_indices": pass_indices,
                    "candidate_inputs_tried": 0,
                    "differential_inputs": 0,
                }
            )
            continue
        candidate_inputs = build_candidate_inputs(
            problem,
            rng=rng,
            max_candidate_inputs=args.max_candidate_inputs,
            max_input_chars=args.max_input_chars,
        )
        behavior_outputs = [
            run_behavior_candidate(
                lcb_repo=Path(args.lcb_repo),
                problem=problem,
                code=code_output,
                behavior_inputs=candidate_inputs,
                timeout=args.behavior_timeout,
            )
            for code_output in code_outputs
        ]
        selected_inputs, selected_indices = choose_differential_inputs(
            behavior_inputs=candidate_inputs,
            behavior_outputs=behavior_outputs,
            candidate_indices=pass_indices,
            max_output_inputs=args.max_output_inputs,
        )
        diagnostics.append(
            {
                "question_id": problem.question_id,
                "question_title": problem.question_title,
                "public_pass_indices": pass_indices,
                "candidate_inputs_tried": len(candidate_inputs),
                "differential_inputs": len(selected_inputs),
                "selected_candidate_input_indices": selected_indices,
                "status": "ok" if selected_inputs else "no_differential_inputs",
            }
        )
        if selected_inputs:
            records.append(
                {
                    "question_id": problem.question_id,
                    "inputs": selected_inputs,
                    "n_inputs": len(selected_inputs),
                    "source": "adaptive_differential_fuzz_v1",
                    "public_pass_indices": pass_indices,
                }
            )

    summary = {
        "source": "adaptive_differential_fuzz_v1",
        "records": len(records),
        "tasks_considered": len(problems),
        "tasks_with_differential_inputs": sum(
            item["differential_inputs"] > 0 for item in diagnostics
        ),
        "candidate_inputs_tried_total": sum(
            item["candidate_inputs_tried"] for item in diagnostics
        ),
        "differential_inputs_total": sum(item["differential_inputs"] for item in diagnostics),
        "seconds": round(time.monotonic() - started, 3),
        "diagnostics": diagnostics,
    }
    return records, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build candidate-aware adaptive differential behavior inputs for LCB "
            "without using hidden-test expected outputs."
        )
    )
    parser.add_argument("--lcb-repo", required=True)
    parser.add_argument("--full-jsonl", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--public-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-candidate-inputs", type=int, default=96)
    parser.add_argument("--max-output-inputs", type=int, default=16)
    parser.add_argument("--max-input-chars", type=int, default=20000)
    parser.add_argument("--behavior-timeout", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument(
        "--hash-full-jsonl",
        action="store_true",
        help="Hash the full JSONL in the manifest. Disabled by default for fast probes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, summary = build_adaptive_records(args)
    behavior_inputs_path = output_dir / "behavior_inputs.json"
    behavior_inputs_path.write_text(
        json.dumps(
            {
                "records": records,
                "metadata": {
                    "source": "adaptive_differential_fuzz_v1",
                    "lcb_repo": args.lcb_repo,
                    "full_jsonl": args.full_jsonl,
                    "full_jsonl_sha256": sha256_file(Path(args.full_jsonl))
                    if args.hash_full_jsonl
                    else None,
                    "generations": args.generations,
                    "generations_sha256": sha256_file(Path(args.generations)),
                    "public_selection": args.public_selection,
                    "public_selection_sha256": sha256_file(Path(args.public_selection)),
                    "max_samples": args.max_samples,
                    "max_candidate_inputs": args.max_candidate_inputs,
                    "max_output_inputs": args.max_output_inputs,
                    "seed": args.seed,
                    "hidden_expected_outputs_used": False,
                },
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
