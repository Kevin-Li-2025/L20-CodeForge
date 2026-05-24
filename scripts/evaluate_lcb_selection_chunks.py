#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import suppress
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_evaluator_module() -> Any:
    script = Path(__file__).with_name("evaluate_lcb_generations.py")
    spec = importlib.util.spec_from_file_location("evaluate_lcb_generations", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVALUATOR = load_evaluator_module()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(EVALUATOR.make_json_safe(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def load_public_selection_records(public_selection_path: Path) -> list[dict[str, Any]]:
    payload = EVALUATOR.load_public_selection_payload(public_selection_path)
    return payload["records"]


def selected_question_ids(public_selection_path: Path) -> list[str]:
    return [
        str(record["question_id"])
        for record in load_public_selection_records(public_selection_path)
    ]


def load_selected_payload(args: argparse.Namespace, question_ids: list[str] | None = None) -> dict[str, Any]:
    lcb_repo = Path(args.lcb_repo)
    generations_path = Path(args.generations)
    public_selection_path = Path(args.public_selection)
    generation_records = EVALUATOR.load_generations(generations_path)
    requested_ids = set(question_ids) if question_ids is not None else set(generation_records)
    problems = EVALUATOR.load_full_jsonl(Path(args.full_jsonl), lcb_repo, requested_ids)

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

    public_selection_payload = EVALUATOR.load_public_selection_payload(public_selection_path)
    final_generations, final_raw_outputs, aligned_records = EVALUATOR.apply_public_selection_records(
        selected_problems,
        generations,
        raw_outputs,
        public_selection_payload["records"],
    )
    return {
        "problems": selected_problems,
        "generations": final_generations,
        "raw_outputs": final_raw_outputs,
        "selection_records": aligned_records,
        "public_selection_payload": public_selection_payload,
    }


def pass_lists_from_results(results: dict[int, list[list[Any]]], total: int) -> list[list[bool]]:
    pass_lists: list[list[bool]] = []
    for index in range(total):
        candidate_results = results.get(index, [])
        pass_lists.append(
            [all(item is True for item in candidate_result) for candidate_result in candidate_results]
        )
    return pass_lists


def metrics_from_pass_lists(pass_lists: list[list[bool]]) -> dict[str, Any]:
    detail = {
        str(index): 1.0 if pass_list and pass_list[0] else 0.0
        for index, pass_list in enumerate(pass_lists)
    }
    total = len(pass_lists)
    passed = sum(value == 1.0 for value in detail.values())
    return {
        "pass@1": passed / total if total else 0.0,
        "detail": {"pass@1": detail},
    }


def evaluate_worker(args: argparse.Namespace) -> dict[str, Any]:
    question_ids = selected_question_ids(Path(args.public_selection))[args.start_index : args.end_index]
    payload = load_selected_payload(args, question_ids)
    problems = payload["problems"]
    final_generations = payload["generations"]
    final_raw_outputs = payload["raw_outputs"]

    _, codegen_metrics = EVALUATOR.load_lcb_modules(Path(args.lcb_repo))
    started = time.monotonic()
    metrics, results, metadata = codegen_metrics(
        [problem.get_evaluation_sample() for problem in problems],
        final_generations,
        k_list=[1],
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
        debug=args.debug,
    )
    evaluation_seconds = round(time.monotonic() - started, 3)

    result_by_index = {int(key): value for key, value in results.items()}
    pass_lists = pass_lists_from_results(result_by_index, len(problems))
    sanitized_metadata = EVALUATOR.sanitize_lcb_metadata(metadata)
    eval_all = [
        problem.insert_output_evaluation(
            output_list=final_raw_outputs[index],
            code_list=final_generations[index],
            graded_list=pass_list,
            metadata=sanitized_metadata[index],
        )
        for index, (problem, pass_list) in enumerate(zip(problems, pass_lists))
    ]
    chunk_payload = {
        "start_index": args.start_index,
        "end_index": args.end_index,
        "question_ids": [problem.question_id for problem in problems],
        "evaluation_seconds": evaluation_seconds,
        "metrics": metrics,
        "results": result_by_index,
        "metadata": sanitized_metadata,
        "pass_lists": pass_lists,
        "eval_all": eval_all,
        "fatal": False,
    }
    write_json(Path(args.chunk_output), chunk_payload)
    return chunk_payload


def fatal_single_payload(args: argparse.Namespace, index: int, log_path: Path) -> dict[str, Any]:
    question_id = selected_question_ids(Path(args.public_selection))[index]
    payload = load_selected_payload(args, [question_id])
    problem = payload["problems"][0]
    final_generations = payload["generations"]
    final_raw_outputs = payload["raw_outputs"]
    metadata = [[
        {
            "error_code": "fatal_chunk_failure",
            "error_message": f"chunk worker failed; see {log_path.name}",
        }
    ]]
    pass_lists = [[False]]
    eval_all = [
        problem.insert_output_evaluation(
            output_list=final_raw_outputs[0],
            code_list=final_generations[0],
            graded_list=pass_lists[0],
            metadata=metadata[0],
        )
    ]
    return {
        "start_index": index,
        "end_index": index + 1,
        "question_ids": [question_id],
        "evaluation_seconds": 0.0,
        "metrics": metrics_from_pass_lists(pass_lists),
        "results": {0: [[False]]},
        "metadata": metadata,
        "pass_lists": pass_lists,
        "eval_all": eval_all,
        "fatal": True,
        "log": str(log_path),
    }


def load_valid_chunk(path: Path, start: int, end: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("start_index") != start or payload.get("end_index") != end:
        return None
    if not isinstance(payload.get("pass_lists"), list):
        return None
    if len(payload["pass_lists"]) != end - start:
        return None
    return payload


def run_worker_subprocess(args: argparse.Namespace, start: int, end: int, chunk_path: Path) -> bool:
    log_path = chunk_path.with_suffix(".log")
    command = [
        sys.executable,
        str(Path(__file__)),
        "--worker",
        "--lcb-repo",
        args.lcb_repo,
        "--lcb-commit",
        args.lcb_commit,
        "--full-jsonl",
        args.full_jsonl,
        "--generations",
        args.generations,
        "--public-selection",
        args.public_selection,
        "--output-dir",
        args.output_dir,
        "--chunk-output",
        str(chunk_path),
        "--start-index",
        str(start),
        "--end-index",
        str(end),
        "--num-process-evaluate",
        str(args.num_process_evaluate),
        "--timeout",
        str(args.timeout),
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.debug:
        command.append("--debug")

    env = os.environ.copy()
    env.setdefault("TMPDIR", "/var/tmp")
    env.setdefault("PYTHONUNBUFFERED", "1")
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, _ = process.communicate(timeout=args.chunk_wall_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, _ = process.communicate()
    elapsed = round(time.monotonic() - started, 3)
    log_path.write_text(
        f"$ {' '.join(command)}\n"
        f"returncode={process.returncode} elapsed_seconds={elapsed} timed_out={timed_out}\n\n"
        + (stdout or ""),
        encoding="utf-8",
    )
    return process.returncode == 0 and load_valid_chunk(chunk_path, start, end) is not None


def evaluate_range(args: argparse.Namespace, start: int, end: int) -> list[dict[str, Any]]:
    chunk_dir = Path(args.output_dir) / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"chunk_{start:04d}_{end:04d}.json"
    cached = load_valid_chunk(chunk_path, start, end)
    if cached is not None and not args.force:
        print(f"reuse chunk {start}:{end}")
        return [cached]

    print(f"evaluate chunk {start}:{end}")
    if run_worker_subprocess(args, start, end, chunk_path):
        return [json.loads(chunk_path.read_text(encoding="utf-8"))]

    if end - start <= 1:
        fatal_payload = fatal_single_payload(args, start, chunk_path.with_suffix(".log"))
        write_json(chunk_path, fatal_payload)
        print(f"fatal single {start}:{end}")
        return [fatal_payload]

    midpoint = start + (end - start) // 2
    return evaluate_range(args, start, midpoint) + evaluate_range(args, midpoint, end)


def remap_results(chunks: list[dict[str, Any]]) -> tuple[dict[int, Any], list[Any], list[list[bool]], list[Any]]:
    results: dict[int, Any] = {}
    metadata: list[Any] = []
    pass_lists: list[list[bool]] = []
    eval_all: list[Any] = []
    for chunk in sorted(chunks, key=lambda item: item["start_index"]):
        start = int(chunk["start_index"])
        for local_index, result in (chunk.get("results") or {}).items():
            results[start + int(local_index)] = result
        metadata.extend(chunk.get("metadata") or [])
        pass_lists.extend(chunk.get("pass_lists") or [])
        eval_all.extend(chunk.get("eval_all") or [])
    return results, metadata, pass_lists, eval_all


def evaluate_manager(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    public_selection_payload = EVALUATOR.load_public_selection_payload(Path(args.public_selection))
    question_ids = [str(record["question_id"]) for record in public_selection_payload["records"]]

    chunks: list[dict[str, Any]] = []
    for start in range(0, len(question_ids), args.chunk_size):
        end = min(start + args.chunk_size, len(question_ids))
        chunks.extend(evaluate_range(args, start, end))

    results, metadata, pass_lists, eval_all = remap_results(chunks)
    metrics = metrics_from_pass_lists(pass_lists)
    evaluation_seconds = round(sum(float(chunk.get("evaluation_seconds", 0.0)) for chunk in chunks), 3)
    fatal_question_ids = [
        question_id
        for chunk in chunks
        if chunk.get("fatal")
        for question_id in chunk.get("question_ids", [])
    ]
    selection_records = public_selection_payload["records"]
    report = {
        "benchmark": "LiveCodeBench code_generation_lite",
        "benchmark_scope": "full release_v6 evaluation from saved generations; chunked hidden-test replay",
        "lcb_repo": args.lcb_repo,
        "lcb_repo_commit": args.lcb_commit,
        "full_jsonl": args.full_jsonl,
        "full_jsonl_sha256": EVALUATOR.sha256_file(Path(args.full_jsonl)),
        "generations": args.generations,
        "generations_sha256": EVALUATOR.sha256_file(Path(args.generations)),
        "tasks_evaluated": len(pass_lists),
        "selection": "public_tests_reused_chunked",
        "public_selection_source": args.public_selection,
        "public_select_tie_breaker": args.public_select_tie_breaker,
        "public_selection_seconds": 0.0,
        "public_selection_metrics": EVALUATOR.make_json_safe(public_selection_payload.get("metrics")),
        "public_selected_public_pass_count": sum(
            record.get("selected_public_score") == 1.0 for record in selection_records
        ),
        "public_oracle_pass_count": sum(bool(record.get("public_oracle_pass")) for record in selection_records),
        "k": [1],
        "max_samples": args.max_samples,
        "chunk_size": args.chunk_size,
        "chunk_wall_timeout": args.chunk_wall_timeout,
        "num_chunks": len(chunks),
        "evaluation_seconds": evaluation_seconds,
        "metrics": metrics,
        "passed_at_1_count": sum(pass_list[0] for pass_list in pass_lists if pass_list),
        "total": len(pass_lists),
        "fatal_chunk_failure_count": len(fatal_question_ids),
        "fatal_question_ids": fatal_question_ids,
    }
    write_json(output_dir / "metrics.json", [metrics, results, metadata])
    write_json(output_dir / "eval_all.json", eval_all)
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved LCB public-selection payload in resumable chunks."
    )
    parser.add_argument("--lcb-repo", required=True)
    parser.add_argument("--lcb-commit", default="unknown")
    parser.add_argument("--full-jsonl", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--public-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--public-select-tie-breaker", default="shortest")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--chunk-wall-timeout", type=int, default=300)
    parser.add_argument("--num-process-evaluate", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--chunk-output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.worker:
        if args.chunk_output is None:
            raise ValueError("--chunk-output is required in --worker mode")
        evaluate_worker(args)
    else:
        evaluate_manager(args)


if __name__ == "__main__":
    main()
