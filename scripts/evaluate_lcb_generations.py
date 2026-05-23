#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def sanitize_lcb_metadata(metadata: Any) -> Any:
    if isinstance(metadata, list):
        return [sanitize_lcb_metadata(item) for item in metadata]
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return {"error_message": metadata[:200]}
    if not isinstance(metadata, dict):
        return {}
    return {
        key: metadata[key]
        for key in ("error_code", "error_message")
        if key in metadata
    }


def get_public_evaluation_sample(problem: Any) -> dict[str, str]:
    return {
        "input_output": json.dumps(
            {
                "inputs": [test.input for test in problem.public_test_cases],
                "outputs": [test.output for test in problem.public_test_cases],
                "fn_name": problem.metadata.get("func_name", None),
            }
        )
    }


def candidate_pass_fraction(result: list[Any]) -> float:
    if not result:
        return 0.0
    return sum(item is True for item in result) / len(result)


def stable_text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def normalize_behavior_value(prefix: str, value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
        except TypeError:
            text = repr(value)
    return f"{prefix}:{len(text)}:{stable_text_hash(text)}"


def bounded_unique(values: list[Any], limit: int) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        try:
            key = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
        except TypeError:
            key = repr(value)
        if key not in seen:
            seen.add(key)
            unique.append(value)
        if len(unique) >= limit:
            break
    return unique


def mutate_json_value(value: Any, limit: int = 6) -> list[Any]:
    variants: list[Any] = []
    if isinstance(value, bool):
        variants.extend([not value])
    elif isinstance(value, int) and not isinstance(value, bool):
        variants.extend([0, 1, -1, value + 1, value - 1, value * 2])
    elif isinstance(value, float):
        variants.extend([0.0, 1.0, -1.0, value + 1.0, value * 2.0])
    elif isinstance(value, str):
        variants.extend(["", value[::-1], value + value[:1], value.upper(), value.lower()])
        if value:
            variants.append(value[1:])
    elif isinstance(value, list):
        variants.extend([[], value[:1], value[:-1], value + value[-1:]])
        for index, item in enumerate(value[:3]):
            for item_variant in mutate_json_value(item, limit=2):
                changed = list(value)
                changed[index] = item_variant
                variants.append(changed)
    elif isinstance(value, dict):
        variants.append({})
        for index, (key, item) in enumerate(list(value.items())[:3]):
            if index >= 3:
                break
            for item_variant in mutate_json_value(item, limit=2):
                changed = dict(value)
                changed[key] = item_variant
                variants.append(changed)
    return bounded_unique([variant for variant in variants if variant != value], limit)


def mutate_functional_input(input_text: str, limit: int) -> list[str]:
    try:
        values = [json.loads(line) for line in input_text.splitlines() if line.strip()]
    except json.JSONDecodeError:
        return []
    variants: list[str] = []
    for value_index, value in enumerate(values):
        for mutated in mutate_json_value(value, limit=limit):
            changed = list(values)
            changed[value_index] = mutated
            variants.append("\n".join(json.dumps(item, ensure_ascii=True) for item in changed))
            if len(variants) >= limit:
                return bounded_unique(variants, limit)
    return bounded_unique(variants, limit)


def mutate_stdin_input(input_text: str, limit: int) -> list[str]:
    import re

    matches = list(re.finditer(r"-?\d+", input_text))
    variants: list[str] = []
    for match in reversed(matches[-8:]):
        original = int(match.group(0))
        for replacement in bounded_unique(
            [0, 1, -1, original + 1, original - 1, original * 2],
            4,
        ):
            if replacement == original:
                continue
            changed = input_text[: match.start()] + str(replacement) + input_text[match.end() :]
            if len(changed) <= 20000:
                variants.append(changed)
            if len(variants) >= limit:
                return bounded_unique(variants, limit)
    return bounded_unique(variants, limit)


def build_behavior_inputs(problem: Any, max_tests: int) -> list[str]:
    inputs: list[str] = []
    for public_test in problem.public_test_cases:
        if public_test.testtype.value == "functional":
            inputs.extend(mutate_functional_input(public_test.input, max_tests))
        else:
            inputs.extend(mutate_stdin_input(public_test.input, max_tests))
        inputs = bounded_unique(inputs, max_tests)
        if len(inputs) >= max_tests:
            break
    return inputs


def _behavior_worker(
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

    outputs: list[str] = []
    try:
        if kind == "functional":
            compiled_sol = compile_code(import_string + "\n\n" + code, timeout)
            method = get_function(compiled_sol, fn_name or "")
            if method is None:
                raise RuntimeError("missing function")
            for behavior_input in behavior_inputs:
                try:
                    args = [json.loads(line) for line in behavior_input.splitlines() if line.strip()]
                    prediction = method(*args)
                    if isinstance(prediction, tuple):
                        prediction = list(prediction)
                    outputs.append(normalize_behavior_value("OK", prediction))
                except BaseException as exc:  # noqa: BLE001 - generated code can raise anything.
                    outputs.append(normalize_behavior_value("ERR", type(exc).__name__))
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
                    outputs.append(normalize_behavior_value("OK", captured_output[0]))
                except BaseException as exc:  # noqa: BLE001 - generated code can raise anything.
                    outputs.append(normalize_behavior_value("ERR", type(exc).__name__))
    except BaseException as exc:  # noqa: BLE001 - generated code can raise anything.
        outputs = [normalize_behavior_value("ERR", type(exc).__name__)] * len(behavior_inputs)
    queue.put(outputs)


def run_behavior_candidate(
    lcb_repo: Path,
    problem: Any,
    code: str,
    behavior_inputs: list[str],
    timeout: int,
) -> list[str]:
    if not behavior_inputs:
        return []
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    kind = "functional" if problem.metadata.get("func_name") else "stdin"
    process = ctx.Process(
        target=_behavior_worker,
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
        return [normalize_behavior_value("ERR", "timeout")] * len(behavior_inputs)
    with contextlib.suppress(Exception):
        return queue.get_nowait()
    return [normalize_behavior_value("ERR", "empty")] * len(behavior_inputs)


def behavior_success_count(outputs: list[str]) -> int:
    return sum(output.startswith("OK:") for output in outputs)


def behavior_consensus_score(candidate_outputs: list[str], all_outputs: list[list[str]]) -> int:
    if not candidate_outputs or not all_outputs:
        return 0
    score = 0
    for test_index, output in enumerate(candidate_outputs):
        counts = Counter(
            outputs[test_index]
            for outputs in all_outputs
            if test_index < len(outputs) and outputs[test_index].startswith("OK:")
        )
        score += counts.get(output, 0)
    return score


def tie_break_candidate_index(
    indices: list[int],
    code_outputs: list[str],
    tie_breaker: str,
) -> int:
    if not indices:
        return 0
    if tie_breaker == "first":
        return indices[0]
    if tie_breaker == "shortest":
        return min(indices, key=lambda index: (len(code_outputs[index]), index))
    if tie_breaker == "longest":
        return max(indices, key=lambda index: (len(code_outputs[index]), -index))
    raise ValueError("tie_breaker must be one of: first, shortest, longest")


def choose_public_selected_index(
    public_results: list[list[Any]],
    code_outputs: list[str],
    tie_breaker: str,
) -> int:
    if not public_results:
        return 0
    scores = [candidate_pass_fraction(result) for result in public_results]
    passing_indices = [index for index, score in enumerate(scores) if score == 1.0]
    if passing_indices:
        return tie_break_candidate_index(passing_indices, code_outputs, tie_breaker)
    best_score = max(scores)
    best_indices = [index for index, score in enumerate(scores) if score == best_score]
    return tie_break_candidate_index(best_indices, code_outputs, tie_breaker)


def build_public_selection_records(
    problems: list[Any],
    generations: list[list[str]],
    public_results: dict[int, list[list[Any]]],
    tie_breaker: str,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    selected_generations: list[list[str]] = []
    records: list[dict[str, Any]] = []
    for problem_index, (problem, code_outputs) in enumerate(zip(problems, generations)):
        problem_public_results = public_results.get(problem_index, [])
        selected_index = choose_public_selected_index(
            public_results=problem_public_results,
            code_outputs=code_outputs,
            tie_breaker=tie_breaker,
        )
        scores = [candidate_pass_fraction(result) for result in problem_public_results]
        public_pass_indices = [index for index, score in enumerate(scores) if score == 1.0]
        selected_generations.append([code_outputs[selected_index] if code_outputs else ""])
        records.append(
            {
                "question_id": problem.question_id,
                "question_title": problem.question_title,
                "selected_index": selected_index,
                "tie_breaker": tie_breaker,
                "n_candidates": len(code_outputs),
                "public_scores": scores,
                "selected_public_score": scores[selected_index] if scores else 0.0,
                "public_pass_indices": public_pass_indices,
                "public_oracle_pass": bool(public_pass_indices),
            }
        )
    return selected_generations, records


def choose_behavior_selected_index(
    public_results: list[list[Any]],
    code_outputs: list[str],
    behavior_outputs: list[list[str]],
    tie_breaker: str,
) -> int:
    public_scores = [candidate_pass_fraction(result) for result in public_results]
    return choose_behavior_selected_index_from_scores(
        public_scores=public_scores,
        code_outputs=code_outputs,
        behavior_outputs=behavior_outputs,
        tie_breaker=tie_breaker,
    )


def choose_behavior_selected_index_from_scores(
    public_scores: list[float],
    code_outputs: list[str],
    behavior_outputs: list[list[str]],
    tie_breaker: str,
) -> int:
    if not code_outputs:
        return 0
    public_scores = list(public_scores[: len(code_outputs)])
    while len(public_scores) < len(code_outputs):
        public_scores.append(0.0)
    signature_counts = Counter(tuple(outputs) for outputs in behavior_outputs)

    def tie_value(index: int) -> int:
        if tie_breaker == "first":
            return -index
        if tie_breaker == "shortest":
            return -len(code_outputs[index])
        if tie_breaker == "longest":
            return len(code_outputs[index])
        raise ValueError("tie_breaker must be one of: first, shortest, longest")

    ranked = []
    for index in range(len(code_outputs)):
        outputs = behavior_outputs[index] if index < len(behavior_outputs) else []
        ranked.append(
            (
                public_scores[index],
                signature_counts.get(tuple(outputs), 0),
                behavior_consensus_score(outputs, behavior_outputs),
                behavior_success_count(outputs),
                tie_value(index),
                -index,
                index,
            )
        )
    return max(ranked)[-1]


def build_behavior_selection_records_from_scores(
    problems: list[Any],
    generations: list[list[str]],
    public_scores_by_problem: dict[int, list[float]],
    behavior_inputs_by_problem: list[list[str]],
    behavior_results: dict[int, list[list[str]]],
    tie_breaker: str,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    selected_generations: list[list[str]] = []
    records: list[dict[str, Any]] = []
    for problem_index, (problem, code_outputs) in enumerate(zip(problems, generations)):
        problem_public_scores = list(public_scores_by_problem.get(problem_index, []))
        problem_behavior_results = behavior_results.get(problem_index, [[] for _ in code_outputs])
        selected_index = choose_behavior_selected_index_from_scores(
            public_scores=problem_public_scores,
            code_outputs=code_outputs,
            behavior_outputs=problem_behavior_results,
            tie_breaker=tie_breaker,
        )
        while len(problem_public_scores) < len(code_outputs):
            problem_public_scores.append(0.0)
        signature_counts = Counter(tuple(outputs) for outputs in problem_behavior_results)
        selected_behavior_outputs = (
            problem_behavior_results[selected_index]
            if selected_index < len(problem_behavior_results)
            else []
        )
        public_pass_indices = [
            index for index, score in enumerate(problem_public_scores) if score == 1.0
        ]
        selected_generations.append([code_outputs[selected_index] if code_outputs else ""])
        records.append(
            {
                "question_id": problem.question_id,
                "question_title": problem.question_title,
                "selected_index": selected_index,
                "tie_breaker": tie_breaker,
                "n_candidates": len(code_outputs),
                "public_scores": problem_public_scores,
                "selected_public_score": problem_public_scores[selected_index]
                if problem_public_scores
                else 0.0,
                "public_pass_indices": public_pass_indices,
                "public_oracle_pass": bool(public_pass_indices),
                "behavior_tests": len(behavior_inputs_by_problem[problem_index]),
                "behavior_signature_hash": stable_text_hash(
                    json.dumps(selected_behavior_outputs, sort_keys=True)
                ),
                "behavior_cluster_size": signature_counts.get(
                    tuple(selected_behavior_outputs),
                    0,
                ),
                "behavior_consensus_score": behavior_consensus_score(
                    selected_behavior_outputs,
                    problem_behavior_results,
                ),
                "behavior_success_count": behavior_success_count(
                    selected_behavior_outputs
                ),
            }
        )
    return selected_generations, records


def build_behavior_selection_records(
    problems: list[Any],
    generations: list[list[str]],
    public_results: dict[int, list[list[Any]]],
    behavior_inputs_by_problem: list[list[str]],
    behavior_results: dict[int, list[list[str]]],
    tie_breaker: str,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    public_scores_by_problem = {
        problem_index: [
            candidate_pass_fraction(result)
            for result in public_results.get(problem_index, [])
        ]
        for problem_index in range(len(problems))
    }
    return build_behavior_selection_records_from_scores(
        problems=problems,
        generations=generations,
        public_scores_by_problem=public_scores_by_problem,
        behavior_inputs_by_problem=behavior_inputs_by_problem,
        behavior_results=behavior_results,
        tie_breaker=tie_breaker,
    )


def public_scores_from_selection_records(
    problems: list[Any],
    generations: list[list[str]],
    selection_records: list[dict[str, Any]],
) -> dict[int, list[float]]:
    records_by_id = {record["question_id"]: record for record in selection_records}
    scores_by_problem: dict[int, list[float]] = {}
    for problem_index, (problem, code_outputs) in enumerate(zip(problems, generations)):
        record = records_by_id.get(problem.question_id)
        if record is None:
            raise ValueError(f"missing public scores for question_id={problem.question_id}")
        raw_scores = record.get("public_scores")
        if not isinstance(raw_scores, list):
            raise ValueError(
                f"public score record for question_id={problem.question_id} is missing public_scores"
            )
        scores = [float(score) for score in raw_scores[: len(code_outputs)]]
        while len(scores) < len(code_outputs):
            scores.append(0.0)
        scores_by_problem[problem_index] = scores
    return scores_by_problem


def load_behavior_input_payload(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        raw_records = payload["records"]
    elif isinstance(payload, list):
        raw_records = payload
    elif isinstance(payload, dict):
        raw_records = [
            {"question_id": question_id, "inputs": inputs}
            for question_id, inputs in payload.items()
        ]
    else:
        raise ValueError("behavior input payload must be a dict, list, or records object")

    inputs_by_id: dict[str, list[str]] = {}
    for record in raw_records:
        if not isinstance(record, dict):
            continue
        question_id = record.get("question_id")
        raw_inputs = record.get("inputs")
        if not isinstance(question_id, str) or not isinstance(raw_inputs, list):
            continue
        inputs = [item for item in raw_inputs if isinstance(item, str) and item.strip()]
        if inputs:
            inputs_by_id[question_id] = inputs
    return inputs_by_id


def load_public_selection_payload(public_selection_path: Path) -> dict[str, Any]:
    payload = json.loads(public_selection_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("public selection payload must contain a records list")
    return payload


def apply_public_selection_records(
    problems: list[Any],
    generations: list[list[str]],
    raw_outputs: list[list[str]],
    selection_records: list[dict[str, Any]],
) -> tuple[list[list[str]], list[list[str]], list[dict[str, Any]]]:
    records_by_id = {record["question_id"]: record for record in selection_records}
    selected_generations: list[list[str]] = []
    selected_raw_outputs: list[list[str]] = []
    aligned_records: list[dict[str, Any]] = []
    for problem, code_outputs, raw_output_list in zip(problems, generations, raw_outputs):
        record = records_by_id.get(problem.question_id)
        if record is None:
            raise ValueError(f"missing public selection for question_id={problem.question_id}")
        selected_index = int(record["selected_index"])
        if selected_index < 0 or selected_index >= len(code_outputs):
            raise ValueError(
                f"selected_index={selected_index} out of range for question_id={problem.question_id}"
            )
        selected_generations.append([code_outputs[selected_index]])
        selected_raw_outputs.append(
            [raw_output_list[selected_index] if selected_index < len(raw_output_list) else ""]
        )
        aligned_records.append(record)
    return selected_generations, selected_raw_outputs, aligned_records


def load_lcb_modules(lcb_repo: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str(lcb_repo))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    return CodeGenerationProblem, codegen_metrics


def load_full_jsonl(
    jsonl: Path,
    lcb_repo: Path,
    question_ids: set[str] | None = None,
) -> list[Any]:
    CodeGenerationProblem, _ = load_lcb_modules(lcb_repo)
    problems = []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if question_ids is not None and record["question_id"] not in question_ids:
                    continue
                problems.append(CodeGenerationProblem(**record))
    return sorted(problems, key=lambda item: item.question_id)


def load_generations(generations_path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(generations_path.read_text(encoding="utf-8"))
    return {record["question_id"]: record for record in records}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    selection_modes = [
        args.public_select,
        bool(args.public_selection),
        args.behavior_select,
        bool(args.behavior_selection),
    ]
    if sum(bool(mode) for mode in selection_modes) > 1:
        raise ValueError(
            "--public-select, --public-selection, --behavior-select, "
            "and --behavior-selection are mutually exclusive"
        )
    if args.behavior_public_scores and not args.behavior_select:
        raise ValueError("--behavior-public-scores can only be used with --behavior-select")
    if args.behavior_inputs and not args.behavior_select:
        raise ValueError("--behavior-inputs can only be used with --behavior-select")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lcb_repo = Path(args.lcb_repo)
    full_jsonl = Path(args.full_jsonl)
    generations_path = Path(args.generations)

    _, codegen_metrics = load_lcb_modules(lcb_repo)
    generation_records = load_generations(generations_path)
    problems = load_full_jsonl(full_jsonl, lcb_repo, set(generation_records))

    selected_problems = []
    generations = []
    raw_outputs = []
    for problem in problems:
        record = generation_records.get(problem.question_id)
        if record is None:
            continue
        code_list = record.get("code_list") or []
        if args.max_samples is not None:
            code_list = code_list[: args.max_samples]
        if not code_list:
            continue
        selected_problems.append(problem)
        generations.append(code_list)
        raw_outputs.append((record.get("raw_outputs") or [])[: len(code_list)])

    if not selected_problems:
        raise ValueError("no overlapping LiveCodeBench generations found")

    public_selection_seconds = 0.0
    public_selection_metrics: dict[str, Any] | None = None
    public_selection_records: list[dict[str, Any]] = []
    public_selection_metadata: Any | None = None
    final_generations = generations
    final_raw_outputs = raw_outputs
    public_selection_source = None
    behavior_inputs_source = None
    if args.public_selection or args.behavior_selection:
        selection_path = Path(args.public_selection or args.behavior_selection)
        public_selection_source = str(selection_path)
        public_selection_payload = load_public_selection_payload(selection_path)
        public_selection_metrics = make_json_safe(public_selection_payload.get("metrics"))
        public_selection_metadata = sanitize_lcb_metadata(public_selection_payload.get("metadata"))
        (
            final_generations,
            final_raw_outputs,
            public_selection_records,
        ) = apply_public_selection_records(
            selected_problems,
            generations,
            raw_outputs,
            public_selection_payload["records"],
        )
    elif args.behavior_select:
        public_started = time.monotonic()
        if args.behavior_public_scores:
            selection_path = Path(args.behavior_public_scores)
            public_selection_source = str(selection_path)
            public_selection_payload = load_public_selection_payload(selection_path)
            public_selection_metrics = make_json_safe(public_selection_payload.get("metrics"))
            public_selection_metadata = sanitize_lcb_metadata(
                public_selection_payload.get("metadata")
            )
            public_scores_by_problem = public_scores_from_selection_records(
                selected_problems,
                generations,
                public_selection_payload["records"],
            )
        else:
            public_samples = [
                get_public_evaluation_sample(problem) for problem in selected_problems
            ]
            public_metrics, public_results, public_metadata = codegen_metrics(
                public_samples,
                generations,
                k_list=[1],
                num_process_evaluate=args.num_process_evaluate,
                timeout=args.public_timeout,
                debug=args.debug,
            )
            public_scores_by_problem = {
                problem_index: [
                    candidate_pass_fraction(result)
                    for result in public_results.get(problem_index, [])
                ]
                for problem_index in range(len(selected_problems))
            }
            public_selection_metrics = make_json_safe(public_metrics)
            public_selection_metadata = sanitize_lcb_metadata(public_metadata)

        external_behavior_inputs: dict[str, list[str]] = {}
        if args.behavior_inputs:
            behavior_inputs_path = Path(args.behavior_inputs)
            behavior_inputs_source = str(behavior_inputs_path)
            external_behavior_inputs = load_behavior_input_payload(behavior_inputs_path)
        behavior_inputs_by_problem = []
        for problem in selected_problems:
            generated_inputs = external_behavior_inputs.get(problem.question_id, [])
            mutation_inputs = build_behavior_inputs(problem, args.max_behavior_tests)
            behavior_inputs_by_problem.append(
                bounded_unique(
                    generated_inputs + mutation_inputs,
                    args.max_behavior_tests,
                )
            )
        behavior_results: dict[int, list[list[str]]] = {}
        for problem_index, (problem, code_outputs) in enumerate(
            zip(selected_problems, generations)
        ):
            problem_behavior_inputs = behavior_inputs_by_problem[problem_index]
            behavior_results[problem_index] = [
                run_behavior_candidate(
                    lcb_repo=lcb_repo,
                    problem=problem,
                    code=code_output,
                    behavior_inputs=problem_behavior_inputs,
                    timeout=args.behavior_timeout,
                )
                for code_output in code_outputs
            ]
        final_generations, public_selection_records = (
            build_behavior_selection_records_from_scores(
                selected_problems,
                generations,
                public_scores_by_problem,
                behavior_inputs_by_problem,
                behavior_results,
                args.public_select_tie_breaker,
            )
        )
        final_raw_outputs = [
            [raw_outputs[index][record["selected_index"]] if raw_outputs[index] else ""]
            for index, record in enumerate(public_selection_records)
        ]
        public_selection_seconds = round(time.monotonic() - public_started, 3)
        (output_dir / "behavior_selection.json").write_text(
            json.dumps(
                {
                    "metrics": public_selection_metrics,
                    "records": public_selection_records,
                    "metadata": public_selection_metadata,
                    "behavior": {
                        "max_behavior_tests": args.max_behavior_tests,
                        "behavior_timeout": args.behavior_timeout,
                        "mutation_version": "v1_public_input_mutations",
                        "behavior_inputs_source": behavior_inputs_source,
                    },
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
    elif args.public_select:
        public_started = time.monotonic()
        public_samples = [get_public_evaluation_sample(problem) for problem in selected_problems]
        public_metrics, public_results, public_metadata = codegen_metrics(
            public_samples,
            generations,
            k_list=[1],
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.public_timeout,
            debug=args.debug,
        )
        final_generations, public_selection_records = build_public_selection_records(
            selected_problems,
            generations,
            public_results,
            args.public_select_tie_breaker,
        )
        final_raw_outputs = [
            [raw_outputs[index][record["selected_index"]] if raw_outputs[index] else ""]
            for index, record in enumerate(public_selection_records)
        ]
        public_selection_seconds = round(time.monotonic() - public_started, 3)
        public_selection_metrics = make_json_safe(public_metrics)
        public_selection_metadata = sanitize_lcb_metadata(public_metadata)
        (output_dir / "public_selection.json").write_text(
            json.dumps(
                {
                    "metrics": public_selection_metrics,
                    "records": public_selection_records,
                    "metadata": public_selection_metadata,
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )

    started = time.monotonic()
    eval_samples = [problem.get_evaluation_sample() for problem in selected_problems]
    metrics, results, metadata = codegen_metrics(
        eval_samples,
        final_generations,
        k_list=args.k,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
        debug=args.debug,
    )
    evaluation_seconds = round(time.monotonic() - started, 3)

    pass_lists = [
        [all(item is True for item in candidate_result) for candidate_result in results[index]]
        for index in range(len(selected_problems))
    ]
    sanitized_metadata = sanitize_lcb_metadata(metadata)
    eval_all = [
        problem.insert_output_evaluation(
            output_list=final_raw_outputs[index],
            code_list=final_generations[index],
            graded_list=pass_list,
            metadata=sanitized_metadata[index],
        )
        for index, (problem, pass_list) in enumerate(zip(selected_problems, pass_lists))
    ]
    report = {
        "benchmark": "LiveCodeBench code_generation_lite",
        "benchmark_scope": "full release_v6 evaluation from saved generations",
        "lcb_repo": str(lcb_repo),
        "lcb_repo_commit": args.lcb_commit,
        "full_jsonl": str(full_jsonl),
        "full_jsonl_sha256": sha256_file(full_jsonl),
        "generations": str(generations_path),
        "generations_sha256": sha256_file(generations_path),
        "tasks_evaluated": len(selected_problems),
        "selection": "public_tests"
        if args.public_select
        else "public_plus_behavior_consensus"
        if args.behavior_select
        else "public_plus_behavior_consensus_reused"
        if args.behavior_selection
        else "public_tests_reused"
        if args.public_selection
        else "none",
        "public_selection_source": public_selection_source,
        "behavior_inputs_source": behavior_inputs_source,
        "public_select_tie_breaker": args.public_select_tie_breaker
        if args.public_select
        or args.public_selection
        or args.behavior_select
        or args.behavior_selection
        else None,
        "public_selection_seconds": public_selection_seconds,
        "public_selection_metrics": public_selection_metrics,
        "public_selected_public_pass_count": sum(
            record["selected_public_score"] == 1.0 for record in public_selection_records
        )
        if args.public_select
        or args.public_selection
        or args.behavior_select
        or args.behavior_selection
        else None,
        "public_oracle_pass_count": sum(
            record["public_oracle_pass"] for record in public_selection_records
        )
        if args.public_select
        or args.public_selection
        or args.behavior_select
        or args.behavior_selection
        else None,
        "behavior_tests_total": sum(
            record.get("behavior_tests", 0) for record in public_selection_records
        )
        if args.behavior_select or args.behavior_selection
        else None,
        "k": args.k,
        "max_samples": args.max_samples,
        "evaluation_seconds": evaluation_seconds,
        "metrics": make_json_safe(metrics),
        "passed_at_1_count": sum(pass_list[0] for pass_list in pass_lists if pass_list),
        "total": len(selected_problems),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(make_json_safe([metrics, results, sanitized_metadata]), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "eval_all.json").write_text(
        json.dumps(make_json_safe(eval_all), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved LCB generations.")
    parser.add_argument("--lcb-repo", required=True)
    parser.add_argument("--lcb-commit", default="unknown")
    parser.add_argument("--full-jsonl", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--public-select", action="store_true")
    parser.add_argument(
        "--public-selection",
        help="Reuse a saved public_selection.json payload instead of re-evaluating public tests.",
    )
    parser.add_argument(
        "--behavior-select",
        action="store_true",
        help=(
            "Select candidates using public tests plus deterministic behavior "
            "consensus on mutated public inputs."
        ),
    )
    parser.add_argument(
        "--behavior-selection",
        help="Reuse a saved behavior_selection.json payload instead of re-running selection.",
    )
    parser.add_argument(
        "--behavior-public-scores",
        help=(
            "When using --behavior-select, reuse public_scores from a saved "
            "public_selection.json or behavior_selection.json instead of re-running public tests."
        ),
    )
    parser.add_argument(
        "--behavior-inputs",
        help=(
            "Optional JSON payload of extra behavior inputs, keyed by question_id or "
            "as records with {question_id, inputs}."
        ),
    )
    parser.add_argument(
        "--public-select-tie-breaker",
        choices=["first", "shortest", "longest"],
        default="shortest",
    )
    parser.add_argument("--public-timeout", type=int, default=6)
    parser.add_argument("--max-behavior-tests", type=int, default=6)
    parser.add_argument("--behavior-timeout", type=int, default=3)
    parser.add_argument("--num-process-evaluate", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
