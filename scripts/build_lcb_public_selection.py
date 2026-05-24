#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any


def load_script_module(name: str, script_name: str) -> Any:
    script = Path(__file__).with_name(script_name)
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LCB_RUNNER = load_script_module("run_lcb_subset_benchmark", "run_lcb_subset_benchmark.py")


def load_generation_records(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list")
    return {
        str(record["question_id"]): record
        for record in records
        if isinstance(record, dict) and record.get("question_id")
    }


def select_problem_generations(
    problems: list[Any],
    generation_records: dict[str, dict[str, Any]],
    max_samples: int | None = None,
) -> tuple[list[Any], list[list[str]]]:
    selected_problems: list[Any] = []
    generations: list[list[str]] = []
    for problem in problems:
        record = generation_records.get(problem.question_id)
        if record is None:
            continue
        code_list = list(record.get("code_list") or [])
        if max_samples is not None:
            code_list = code_list[:max_samples]
        if not code_list:
            continue
        selected_problems.append(problem)
        generations.append(code_list)
    return selected_problems, generations


def public_k_list_for_generations(generations: list[list[str]]) -> list[int]:
    max_candidates = max((len(items) for items in generations), default=1)
    return sorted({1, max_candidates})


def build_public_selection(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    lcb_repo = Path(args.lcb_repo)
    parquet_path = Path(args.parquet)
    generations_path = Path(args.generations)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _, codegen_metrics = LCB_RUNNER.load_lcb_modules(lcb_repo)
    problems = LCB_RUNNER.load_problems_from_parquet(
        parquet_path=parquet_path,
        lcb_repo=lcb_repo,
        start_date=LCB_RUNNER.parse_iso_date(args.start_date),
        end_date=LCB_RUNNER.parse_iso_date(args.end_date),
        difficulty=args.difficulty,
        limit=args.limit,
    )
    generation_records = load_generation_records(generations_path)
    selected_problems, generations = select_problem_generations(
        problems=problems,
        generation_records=generation_records,
        max_samples=args.max_samples,
    )
    if not selected_problems:
        raise ValueError("no overlapping LiveCodeBench generations found")

    public_samples = [
        LCB_RUNNER.get_public_evaluation_sample(problem) for problem in selected_problems
    ]
    public_metrics, public_results, public_metadata = codegen_metrics(
        public_samples,
        generations,
        k_list=public_k_list_for_generations(generations),
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
        debug=args.debug,
    )
    _, public_selection_records = LCB_RUNNER.build_public_selection_records(
        problems=selected_problems,
        generations=generations,
        public_results=public_results,
        tie_breaker=args.tie_breaker,
    )
    payload = {
        "metrics": LCB_RUNNER.make_json_safe(public_metrics),
        "records": public_selection_records,
        "metadata": LCB_RUNNER.sanitize_lcb_metadata(public_metadata),
        "report": {
            "benchmark": "LiveCodeBench code_generation_lite",
            "benchmark_scope": "public-test selection only; hidden tests not evaluated",
            "lcb_repo": str(lcb_repo),
            "lcb_commit": args.lcb_commit,
            "parquet": str(parquet_path),
            "parquet_sha256": LCB_RUNNER.sha256_file(parquet_path),
            "generations": str(generations_path),
            "generations_sha256": LCB_RUNNER.sha256_file(generations_path),
            "tasks_selected": len(selected_problems),
            "max_samples": args.max_samples,
            "tie_breaker": args.tie_breaker,
            "public_selected_pass_count": sum(
                record["selected_public_score"] == 1.0
                for record in public_selection_records
            ),
            "public_oracle_pass_count": sum(
                record["public_oracle_pass"] for record in public_selection_records
            ),
            "seconds": round(time.monotonic() - started, 3),
        },
    }
    output_path.write_text(
        json.dumps(LCB_RUNNER.make_json_safe(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["report"], indent=2, ensure_ascii=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reusable LiveCodeBench public_selection.json payload from saved "
            "generations without evaluating hidden tests."
        )
    )
    parser.add_argument("--lcb-repo", required=True)
    parser.add_argument("--lcb-commit", default="unknown")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--tie-breaker",
        choices=["first", "shortest", "longest"],
        default="shortest",
    )
    parser.add_argument("--num-process-evaluate", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    build_public_selection(build_parser().parse_args())


if __name__ == "__main__":
    main()
