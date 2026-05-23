#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_lcb_generations import (  # noqa: E402
    load_full_jsonl,
    load_generations,
    load_public_selection_payload,
    sha256_file,
)


def enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n# ... truncated ..."


def load_problem_records(full_jsonl: Path, lcb_repo: Path, question_ids: set[str]) -> dict[str, Any]:
    return {
        problem.question_id: problem
        for problem in load_full_jsonl(full_jsonl, lcb_repo, question_ids)
    }


def public_signal_features(record: dict[str, Any]) -> dict[str, Any]:
    scores = [float(score) for score in record.get("public_scores", [])]
    best_score = max(scores) if scores else 0.0
    return {
        "best_public_score": best_score,
        "best_public_tie_count": sum(score == best_score for score in scores),
        "public_pass_count": sum(score == 1.0 for score in scores),
        "partial_public_score_count": sum(0.0 < score < 1.0 for score in scores),
        "nonzero_public_score_count": sum(score > 0.0 for score in scores),
        "public_score_sum": round(sum(scores), 6),
    }


def target_priority_key(
    record: dict[str, Any],
    original_index: int,
    target_priority: str,
) -> tuple[Any, ...]:
    features = public_signal_features(record)
    if target_priority == "input-order":
        return (original_index,)
    if target_priority == "public-ambiguity":
        return (
            -features["best_public_score"],
            -features["best_public_tie_count"],
            -features["public_pass_count"],
            -features["partial_public_score_count"],
            original_index,
        )
    if target_priority == "public-fragility":
        return (
            -features["best_public_score"],
            features["public_pass_count"],
            -features["partial_public_score_count"],
            features["public_score_sum"],
            original_index,
        )
    raise ValueError(
        "target_priority must be one of: input-order, public-ambiguity, public-fragility"
    )


def select_target_records(
    records: list[dict[str, Any]],
    min_public_score: float,
    include_non_ties: bool,
    limit: int | None,
    target_priority: str = "input-order",
) -> list[dict[str, Any]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for original_index, record in enumerate(records):
        features = public_signal_features(record)
        if not record.get("public_scores"):
            continue
        if features["best_public_score"] < min_public_score:
            continue
        if not include_non_ties and features["best_public_tie_count"] < 2:
            continue
        selected.append((original_index, record))
    selected.sort(
        key=lambda item: target_priority_key(
            record=item[1],
            original_index=item[0],
            target_priority=target_priority,
        )
    )
    if limit is not None:
        selected = selected[:limit]
    return [record for _, record in selected]


def public_test_payload(problem: Any, max_public_tests: int) -> list[dict[str, str]]:
    tests = []
    for test in problem.public_test_cases[:max_public_tests]:
        tests.append({"input": test.input, "output": test.output})
    return tests


def build_prompt(
    problem: Any,
    code_outputs: list[str],
    public_scores: list[float],
    candidate_indices: list[int],
    max_code_chars: int,
    max_public_tests: int,
) -> str:
    fn_name = problem.metadata.get("func_name")
    interface = (
        "functional. Each generated input must be a single string containing one JSON value "
        "per argument, separated by newlines, matching the function signature."
        if fn_name
        else "stdin/stdout. Each generated input must be the exact raw stdin string."
    )
    candidate_blocks = []
    for index in candidate_indices:
        if index >= len(code_outputs):
            continue
        candidate_blocks.append(
            f"### Candidate {index} public_score={public_scores[index]:.3f}\n"
            f"```python\n{truncate_text(code_outputs[index], max_code_chars)}\n```"
        )
    public_tests = json.dumps(
        public_test_payload(problem, max_public_tests),
        ensure_ascii=True,
        indent=2,
    )
    return (
        "You are generating adversarial differential tests for a programming-contest "
        "problem. Use only the problem statement, public examples, and candidate code. "
        "Do not assume any hidden tests.\n\n"
        "Goal: create inputs that distinguish plausible wrong solutions from correct "
        "solutions. Prefer edge cases, boundary sizes, parity/sign corner cases, empty "
        "or singleton structures, duplicated values, already-sorted/reversed orders, "
        "and cases where greedy/local reasoning can fail.\n\n"
        f"Interface: {interface}\n"
        f"Function name, if functional: {fn_name or ''}\n\n"
        f"### Problem\n{problem.question_content}\n\n"
        f"### Public tests\n{public_tests}\n\n"
        f"{chr(10).join(candidate_blocks)}\n\n"
        "Return JSON only, with this exact schema:\n"
        "{\"inputs\": [\"...\"]}\n"
        "Do not include expected outputs. Do not include explanations. "
        "Use at most 12 inputs."
    )


def extract_json_payload(text: str) -> Any:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", stripped):
        candidate = stripped[match.start() :]
        for repair in (False, True):
            with contextlib.suppress(json.JSONDecodeError):
                repaired = repair_invalid_json_escapes(candidate) if repair else candidate
                value, _ = decoder.raw_decode(repaired)
                return value
    raise ValueError("could not parse JSON payload")


def repair_invalid_json_escapes(text: str) -> str:
    """Keep JSON valid when a model emits raw backslashes inside strings."""
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if next_char and next_char in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
                result.append(char)
            else:
                result.append("\\\\")
            index += 1
            continue
        result.append(char)
        index += 1
    return "".join(result)


def coerce_inputs(value: Any, max_inputs: int, max_input_chars: int) -> list[str]:
    raw_inputs = value.get("inputs") if isinstance(value, dict) else value
    if not isinstance(raw_inputs, list):
        return []
    inputs: list[str] = []
    seen: set[str] = set()
    for item in raw_inputs:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or len(text) > max_input_chars or text in seen:
            continue
        seen.add(text)
        inputs.append(text)
        if len(inputs) >= max_inputs:
            break
    return inputs


def parse_llm_outputs(
    llm_output_jsonl: Path,
    max_inputs: int,
    max_input_chars: int,
) -> list[dict[str, Any]]:
    records = []
    with llm_output_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = row.get("question_id")
            text = row.get("text") or row.get("output") or row.get("raw_output")
            if not isinstance(question_id, str) or not isinstance(text, str):
                continue
            try:
                inputs = coerce_inputs(
                    extract_json_payload(text),
                    max_inputs=max_inputs,
                    max_input_chars=max_input_chars,
                )
            except ValueError:
                inputs = []
            if inputs:
                records.append(
                    {
                        "question_id": question_id,
                        "inputs": inputs,
                        "source": "llm_candidate_aware_differential_v1",
                    }
                )
    return records


def build_prompt_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    selection_payload = load_public_selection_payload(Path(args.public_selection))
    target_records = select_target_records(
        selection_payload["records"],
        min_public_score=args.min_public_score,
        include_non_ties=args.include_non_ties,
        limit=args.limit,
        target_priority=args.target_priority,
    )
    generations = load_generations(Path(args.generations))
    problems_by_id = load_problem_records(
        Path(args.full_jsonl),
        Path(args.lcb_repo),
        {record["question_id"] for record in target_records},
    )
    prompt_records = []
    for record in target_records:
        question_id = record["question_id"]
        problem = problems_by_id.get(question_id)
        generation = generations.get(question_id)
        if problem is None or generation is None:
            continue
        code_outputs = generation.get("code_list") or []
        if args.max_samples is not None:
            code_outputs = code_outputs[: args.max_samples]
        public_scores = [float(score) for score in record.get("public_scores", [])]
        while len(public_scores) < len(code_outputs):
            public_scores.append(0.0)
        best_score = max(public_scores) if public_scores else 0.0
        candidate_indices = [
            index for index, score in enumerate(public_scores) if score == best_score
        ]
        prompt = build_prompt(
            problem=problem,
            code_outputs=code_outputs,
            public_scores=public_scores,
            candidate_indices=candidate_indices,
            max_code_chars=args.max_code_chars,
            max_public_tests=args.max_public_tests,
        )
        prompt_records.append(
            {
                "question_id": question_id,
                "question_title": problem.question_title,
                "difficulty": enum_value(getattr(problem, "difficulty", "")),
                "platform": enum_value(getattr(problem, "platform", "")),
                "func_name": problem.metadata.get("func_name"),
                "public_scores": public_scores,
                "public_signal_features": public_signal_features(record),
                "target_priority": args.target_priority,
                "candidate_indices": candidate_indices,
                "prompt_sha256_16": stable_hash(prompt),
                "prompt": prompt,
            }
        )
    return prompt_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build candidate-aware LCB behavior-test prompts and parse LLM outputs."
    )
    parser.add_argument("--lcb-repo", required=True)
    parser.add_argument("--full-jsonl", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--public-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--min-public-score", type=float, default=1.0)
    parser.add_argument("--include-non-ties", action="store_true")
    parser.add_argument(
        "--target-priority",
        choices=["input-order", "public-ambiguity", "public-fragility"],
        default="input-order",
        help=(
            "Ordering for the finite prompt budget. public-fragility prioritizes "
            "public-passing ties with fewer public-passing candidates and more "
            "partial public failures; it uses public scores only."
        ),
    )
    parser.add_argument("--max-code-chars", type=int, default=5000)
    parser.add_argument("--max-public-tests", type=int, default=3)
    parser.add_argument("--llm-output-jsonl")
    parser.add_argument("--max-inputs", type=int, default=12)
    parser.add_argument("--max-input-chars", type=int, default=20000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_records = build_prompt_records(args)
    prompts_path = output_dir / "prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as handle:
        for record in prompt_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    parsed_records: list[dict[str, Any]] = []
    if args.llm_output_jsonl:
        parsed_records = parse_llm_outputs(
            Path(args.llm_output_jsonl),
            max_inputs=args.max_inputs,
            max_input_chars=args.max_input_chars,
        )
        (output_dir / "behavior_inputs.json").write_text(
            json.dumps(
                {
                    "records": parsed_records,
                    "metadata": {
                        "source": "llm_candidate_aware_differential_v1",
                        "llm_output_jsonl": args.llm_output_jsonl,
                    },
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )

    manifest = {
        "prompts": str(prompts_path),
        "prompt_count": len(prompt_records),
        "parsed_behavior_input_count": len(parsed_records),
        "lcb_repo": args.lcb_repo,
        "full_jsonl": args.full_jsonl,
        "full_jsonl_sha256": sha256_file(Path(args.full_jsonl)),
        "generations": args.generations,
        "generations_sha256": sha256_file(Path(args.generations)),
        "public_selection": args.public_selection,
        "public_selection_sha256": sha256_file(Path(args.public_selection)),
        "targeting": {
            "min_public_score": args.min_public_score,
            "include_non_ties": args.include_non_ties,
            "target_priority": args.target_priority,
            "limit": args.limit,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
