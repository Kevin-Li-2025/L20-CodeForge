#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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


def load_lcb_modules(lcb_repo: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str(lcb_repo))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    return CodeGenerationProblem, codegen_metrics


def load_full_jsonl(jsonl: Path, lcb_repo: Path) -> list[Any]:
    CodeGenerationProblem, _ = load_lcb_modules(lcb_repo)
    problems = []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                problems.append(CodeGenerationProblem(**json.loads(line)))
    return sorted(problems, key=lambda item: item.question_id)


def load_generations(generations_path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(generations_path.read_text(encoding="utf-8"))
    return {record["question_id"]: record for record in records}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lcb_repo = Path(args.lcb_repo)
    full_jsonl = Path(args.full_jsonl)
    generations_path = Path(args.generations)

    _, codegen_metrics = load_lcb_modules(lcb_repo)
    problems = load_full_jsonl(full_jsonl, lcb_repo)
    generation_records = load_generations(generations_path)

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
    if args.public_select:
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
        "selection": "public_tests" if args.public_select else "none",
        "public_select_tie_breaker": args.public_select_tie_breaker
        if args.public_select
        else None,
        "public_selection_seconds": public_selection_seconds,
        "public_selection_metrics": public_selection_metrics,
        "public_selected_public_pass_count": sum(
            record["selected_public_score"] == 1.0 for record in public_selection_records
        )
        if args.public_select
        else None,
        "public_oracle_pass_count": sum(
            record["public_oracle_pass"] for record in public_selection_records
        )
        if args.public_select
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
        "--public-select-tie-breaker",
        choices=["first", "shortest", "longest"],
        default="shortest",
    )
    parser.add_argument("--public-timeout", type=int, default=6)
    parser.add_argument("--num-process-evaluate", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
