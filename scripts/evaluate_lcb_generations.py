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

    started = time.monotonic()
    eval_samples = [problem.get_evaluation_sample() for problem in selected_problems]
    metrics, results, metadata = codegen_metrics(
        eval_samples,
        generations,
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
            output_list=raw_outputs[index],
            code_list=generations[index],
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
    parser.add_argument("--num-process-evaluate", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
