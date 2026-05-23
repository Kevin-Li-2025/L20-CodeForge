#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lcb_behavior_test_prompts import extract_json_payload, sha256_file, truncate_text  # noqa: E402
from evaluate_lcb_generations import (  # noqa: E402
    load_behavior_input_payload,
    load_full_jsonl,
    load_generations,
    load_public_selection_payload,
    make_json_safe,
    normalize_behavior_value,
    stable_text_hash,
)


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def prompt_id(question_id: str, input_index: int) -> str:
    digest = hashlib.sha256(f"{question_id}:{input_index}".encode()).hexdigest()[:12]
    return f"{question_id}:{input_index}:{digest}"


def label_for_index(index: int) -> str:
    if index < len(LABELS):
        return LABELS[index]
    return f"OPTION_{index + 1}"


def coerce_public_scores(record: dict[str, Any], n_candidates: int) -> list[float]:
    scores = [float(score) for score in record.get("public_scores", [])[:n_candidates]]
    while len(scores) < n_candidates:
        scores.append(0.0)
    return scores


def raw_result_display(result: dict[str, Any], max_chars: int) -> str:
    if result["status"] != "OK":
        return f"<{result['status']}>"
    value = result.get("value")
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    return truncate_text(text, max_chars)


def normalize_raw_result(result: dict[str, Any]) -> str:
    if result["status"] == "OK":
        return normalize_behavior_value("OK", result.get("value"))
    return normalize_behavior_value("ERR", result.get("error", result["status"]))


def _raw_behavior_worker(
    lcb_repo: str,
    kind: str,
    fn_name: str | None,
    code: str,
    behavior_inputs: list[str],
    timeout: int,
    queue: Any,
) -> None:
    sys.path.insert(0, lcb_repo)
    from lcb_runner.evaluation.testing_util import (
        Capturing,
        call_method,
        clean_if_name,
        compile_code,
        get_function,
        import_string,
        make_function,
    )

    results: list[dict[str, Any]] = []
    try:
        if kind == "functional":
            compiled_sol = compile_code(import_string + "\n\n" + code, timeout)
            method = get_function(compiled_sol, fn_name or "")
            if method is None:
                raise RuntimeError("missing function")
            for behavior_input in behavior_inputs:
                try:
                    args = [
                        json.loads(line)
                        for line in behavior_input.splitlines()
                        if line.strip()
                    ]
                    prediction = method(*args)
                    if isinstance(prediction, tuple):
                        prediction = list(prediction)
                    results.append({"status": "OK", "value": make_json_safe(prediction)})
                except BaseException as exc:  # noqa: BLE001 - generated code can raise anything.
                    results.append({"status": "ERR", "error": type(exc).__name__})
        else:
            wrapped_code = make_function(clean_if_name(code))
            compiled_sol = compile_code(wrapped_code, timeout)
            method = get_function(compiled_sol, "wrapped_function")
            if method is None:
                raise RuntimeError("missing wrapped function")
            for behavior_input in behavior_inputs:
                try:
                    with Capturing() as captured_output:
                        call_method(method, behavior_input)
                    results.append({"status": "OK", "value": captured_output[0]})
                except BaseException as exc:  # noqa: BLE001 - generated code can raise anything.
                    results.append({"status": "ERR", "error": type(exc).__name__})
    except BaseException as exc:  # noqa: BLE001 - generated code can raise anything.
        results = [{"status": "ERR", "error": type(exc).__name__}] * len(behavior_inputs)
    queue.put(results)


def run_raw_behavior_candidate(
    lcb_repo: Path,
    problem: Any,
    code: str,
    behavior_inputs: list[str],
    timeout: int,
) -> list[dict[str, Any]]:
    if not behavior_inputs:
        return []
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    kind = "functional" if problem.metadata.get("func_name") else "stdin"
    process = ctx.Process(
        target=_raw_behavior_worker,
        args=(
            str(lcb_repo),
            kind,
            problem.metadata.get("func_name"),
            code,
            behavior_inputs,
            timeout,
            queue,
        ),
    )
    process.start()
    process.join((timeout + 1) * len(behavior_inputs) + 3)
    if process.is_alive():
        process.kill()
        process.join()
        return [{"status": "ERR", "error": "timeout"}] * len(behavior_inputs)
    with contextlib.suppress(Exception):
        return queue.get_nowait()
    return [{"status": "ERR", "error": "empty"}] * len(behavior_inputs)


def build_output_options(
    candidate_results_for_input: list[dict[str, Any]],
    max_option_chars: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for candidate_index, result in enumerate(candidate_results_for_input):
        if result["status"] != "OK":
            continue
        normalized = normalize_raw_result(result)
        if normalized not in grouped:
            grouped[normalized] = {
                "normalized_output": normalized,
                "value": result.get("value"),
                "display": raw_result_display(result, max_option_chars),
                "candidate_indices": [],
            }
        grouped[normalized]["candidate_indices"].append(candidate_index)
    options = []
    for option_index, option in enumerate(grouped.values()):
        options.append({"label": label_for_index(option_index), **option})
    return options


def public_test_payload(problem: Any, max_public_tests: int) -> list[dict[str, str]]:
    tests = []
    for test in problem.public_test_cases[:max_public_tests]:
        tests.append({"input": test.input, "output": test.output})
    return tests


def build_prompt(
    problem: Any,
    behavior_input: str,
    options: list[dict[str, Any]],
    max_problem_chars: int,
    max_public_tests: int,
) -> str:
    fn_name = problem.metadata.get("func_name")
    interface = (
        "functional. The input contains one JSON value per function argument, "
        "separated by newlines. Choose the function return value."
        if fn_name
        else "stdin/stdout. Choose the exact stdout for the raw stdin input."
    )
    options_text = "\n".join(
        f"Option {option['label']}:\n```text\n{option['display']}\n```\n"
        f"Produced by candidates: {option['candidate_indices']}"
        for option in options
    )
    public_tests = json.dumps(
        public_test_payload(problem, max_public_tests),
        ensure_ascii=True,
        indent=2,
    )
    return (
        "You are an expected-output verifier for a programming-contest problem. "
        "Use only the problem statement, the public examples, and the single input "
        "below. Do not assume or mention hidden tests.\n\n"
        "Task: determine which candidate output option is correct for the input. "
        "If none of the options is correct, choose NONE. Return JSON only.\n\n"
        f"Interface: {interface}\n"
        f"Function name, if functional: {fn_name or ''}\n\n"
        f"### Problem\n{truncate_text(problem.question_content, max_problem_chars)}\n\n"
        f"### Public examples\n{public_tests}\n\n"
        f"### Input to verify\n```text\n{behavior_input}\n```\n\n"
        f"### Candidate output options\n{options_text}\n\n"
        "Return exactly this schema:\n"
        "{\"choice\": \"A|B|C|NONE\", \"confidence\": 0.0, \"reason\": \"short\"}"
    )


def parse_choice_payload(text: str) -> dict[str, Any]:
    payload = extract_json_payload(text)
    if not isinstance(payload, dict):
        raise ValueError("verifier output must be a JSON object")
    raw_choice = str(payload.get("choice") or "").strip().upper()
    match = re.search(r"\b([A-Z]|NONE|OPTION_\d+)\b", raw_choice)
    choice = match.group(1) if match else "NONE"
    if choice == "NO":
        choice = "NONE"
    confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "choice": choice,
        "confidence": confidence,
        "reason": str(payload.get("reason") or "")[:500],
    }


def parse_llm_outputs(
    output_jsonl: Path,
    prompt_records_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    parsed = []
    with output_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("record_id") or row.get("prompt_id")
            raw_output = row.get("raw_output") or row.get("text") or row.get("output")
            if not isinstance(record_id, str) or not isinstance(raw_output, str):
                continue
            prompt_record = prompt_records_by_id.get(record_id)
            if prompt_record is None:
                continue
            with contextlib.suppress(ValueError):
                choice = parse_choice_payload(raw_output)
                parsed.append(
                    {
                        "record_id": record_id,
                        "question_id": prompt_record["question_id"],
                        "input_index": prompt_record["input_index"],
                        **choice,
                    }
                )
    return parsed


def build_prompt_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.monotonic()
    behavior_inputs_by_id = load_behavior_input_payload(Path(args.behavior_inputs))
    question_ids = set(behavior_inputs_by_id)
    generation_records = load_generations(Path(args.generations))
    public_selection_payload = load_public_selection_payload(Path(args.public_selection))
    selection_by_id = {
        record["question_id"]: record for record in public_selection_payload["records"]
    }
    problems = load_full_jsonl(Path(args.full_jsonl), Path(args.lcb_repo), question_ids)
    if args.limit is not None:
        problems = problems[: args.limit]

    prompt_records: list[dict[str, Any]] = []
    candidate_output_records: list[dict[str, Any]] = []
    skipped = 0
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
        public_scores = coerce_public_scores(selection_record, len(code_outputs))
        public_pass_indices = [
            index for index, score in enumerate(public_scores) if score == 1.0
        ]
        behavior_inputs = behavior_inputs_by_id.get(problem.question_id, [])[
            : args.max_inputs_per_task
        ]
        print(
            f"[{problem_number}/{len(problems)}] {problem.question_id} "
            f"inputs={len(behavior_inputs)}",
            flush=True,
        )
        raw_results_by_candidate = [
            run_raw_behavior_candidate(
                lcb_repo=Path(args.lcb_repo),
                problem=problem,
                code=code_output,
                behavior_inputs=behavior_inputs,
                timeout=args.behavior_timeout,
            )
            for code_output in code_outputs
        ]
        for input_index, behavior_input in enumerate(behavior_inputs):
            candidate_results = [
                raw_results[input_index]
                if input_index < len(raw_results)
                else {"status": "ERR", "error": "missing"}
                for raw_results in raw_results_by_candidate
            ]
            options = build_output_options(candidate_results, args.max_option_chars)
            candidate_output_records.append(
                {
                    "question_id": problem.question_id,
                    "input_index": input_index,
                    "input_sha256_16": stable_text_hash(behavior_input),
                    "candidate_results": candidate_results,
                    "options": options,
                }
            )
            if len(options) < args.min_options:
                skipped += 1
                continue
            record_id = prompt_id(problem.question_id, input_index)
            prompt = build_prompt(
                problem=problem,
                behavior_input=behavior_input,
                options=options,
                max_problem_chars=args.max_problem_chars,
                max_public_tests=args.max_public_tests,
            )
            prompt_records.append(
                {
                    "record_id": record_id,
                    "question_id": problem.question_id,
                    "question_title": problem.question_title,
                    "input_index": input_index,
                    "input_sha256_16": stable_text_hash(behavior_input),
                    "public_pass_indices": public_pass_indices,
                    "public_scores": public_scores,
                    "options": options,
                    "prompt_sha256_16": stable_text_hash(prompt),
                    "prompt": prompt,
                }
            )

    metadata = {
        "source": "lcb_expected_output_choice_verifier_v1",
        "prompt_count": len(prompt_records),
        "candidate_output_records": len(candidate_output_records),
        "skipped_less_than_min_options": skipped,
        "seconds": round(time.monotonic() - started, 3),
        "hidden_expected_outputs_used": False,
    }
    return prompt_records, {"metadata": metadata, "candidate_outputs": candidate_output_records}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build expected-output choice verifier prompts for LCB differential inputs."
    )
    parser.add_argument("--lcb-repo", required=True)
    parser.add_argument("--full-jsonl", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--public-selection", required=True)
    parser.add_argument("--behavior-inputs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-inputs-per-task", type=int, default=8)
    parser.add_argument("--min-options", type=int, default=2)
    parser.add_argument("--behavior-timeout", type=int, default=1)
    parser.add_argument("--max-problem-chars", type=int, default=6000)
    parser.add_argument("--max-option-chars", type=int, default=1200)
    parser.add_argument("--max-public-tests", type=int, default=3)
    parser.add_argument("--llm-output-jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_records, candidate_payload = build_prompt_records(args)
    prompts_path = output_dir / "prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as handle:
        for record in prompt_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    candidate_payload["metadata"].update(
        {
            "prompts": str(prompts_path),
            "behavior_inputs": args.behavior_inputs,
            "behavior_inputs_sha256": sha256_file(Path(args.behavior_inputs)),
            "generations": args.generations,
            "generations_sha256": sha256_file(Path(args.generations)),
            "public_selection": args.public_selection,
            "public_selection_sha256": sha256_file(Path(args.public_selection)),
        }
    )
    (output_dir / "candidate_outputs.json").write_text(
        json.dumps(candidate_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if args.llm_output_jsonl:
        parsed = parse_llm_outputs(
            Path(args.llm_output_jsonl),
            {record["record_id"]: record for record in prompt_records},
        )
        (output_dir / "verifier_choices.json").write_text(
            json.dumps(
                {
                    "records": parsed,
                    "metadata": {
                        "source": "lcb_expected_output_choice_verifier_v1",
                        "llm_output_jsonl": args.llm_output_jsonl,
                    },
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(candidate_payload["metadata"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
