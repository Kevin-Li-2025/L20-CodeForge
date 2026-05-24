#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any


SECOND_PASS_SYSTEM_MESSAGE = (
    "You convert reasoning-heavy coding attempts into final executable Python "
    "solutions. Return only final code."
)


def load_script_module(name: str, script_name: str) -> Any:
    script = Path(__file__).with_name(script_name)
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_script_module("run_lcb_subset_benchmark", "run_lcb_subset_benchmark.py")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_records(path: Path) -> dict[str, dict[str, Any]]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError("source generations must be a JSON list")
    return {
        str(record["question_id"]): record
        for record in records
        if isinstance(record, dict) and record.get("question_id") is not None
    }


def parse_candidate_indices(value: str | None) -> list[int] | None:
    if not value:
        return None
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("--source-candidate-indices must contain at least one index")
    if any(index < 0 for index in indices):
        raise ValueError("--source-candidate-indices must be non-negative")
    return indices


def select_source_texts(
    record: dict[str, Any],
    field: str,
    max_source_candidates: int,
    candidate_indices: list[int] | None = None,
) -> list[tuple[int, str]]:
    source_values = record.get(field) or []
    if not isinstance(source_values, list):
        return []
    indexed = [
        (index, value)
        for index, value in enumerate(source_values)
        if isinstance(value, str) and value.strip()
    ]
    if candidate_indices is not None:
        wanted = set(candidate_indices)
        indexed = [(index, value) for index, value in indexed if index in wanted]
    return indexed[:max_source_candidates]


def truncate_reasoning(text: str, max_chars: int) -> str:
    stripped = text.strip()
    if max_chars <= 0 or len(stripped) <= max_chars:
        return stripped
    return stripped[-max_chars:].lstrip()


def build_second_pass_prompt(
    problem: Any,
    source_text: str,
    reasoning_max_chars: int,
) -> str:
    starter_code = getattr(problem, "starter_code", "") or ""
    reasoning = truncate_reasoning(source_text, reasoning_max_chars)
    prompt = [
        "### Original problem",
        problem.question_content,
        "",
    ]
    if starter_code:
        prompt.extend(
            [
                "### Required starter signature",
                "```python",
                starter_code.strip(),
                "```",
                "",
            ]
        )
    prompt.extend(
        [
            "### Previous attempt or reasoning",
            reasoning,
            "",
            "### Task",
            "Using the original problem and the useful ideas above, write the final "
            "complete Python solution.",
            "Do not think aloud. Do not include analysis, examples, markdown prose, "
            "or comments-only bodies.",
            "The answer must be syntactically valid and must use the required starter "
            "signature when one is provided.",
            "Return exactly one fenced Python code block.",
            "",
            "### Final answer",
        ]
    )
    return "\n".join(prompt).strip() + "\n"


def build_generation_record(
    problem: Any,
    prompt: str,
    source_generations: str,
    source_candidate_indices: list[int],
    raw_outputs: list[str],
    code_outputs: list[str],
) -> dict[str, Any]:
    return {
        "question_id": problem.question_id,
        "question_title": problem.question_title,
        "contest_date": problem.contest_date.isoformat(),
        "platform": problem.platform.value,
        "difficulty": problem.difficulty.value,
        "prompt": prompt,
        "source_generations": source_generations,
        "source_candidate_indices": source_candidate_indices,
        "raw_outputs": raw_outputs,
        "code_list": code_outputs,
    }


def regenerate(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / "generations.json"
    if any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")

    source_generations_path = Path(args.source_generations)
    source_records = load_source_records(source_generations_path)
    candidate_indices = parse_candidate_indices(args.source_candidate_indices)
    problems = RUNNER.load_problems_from_parquet(
        parquet_path=Path(args.parquet),
        lcb_repo=Path(args.lcb_repo),
        question_ids=RUNNER.parse_question_ids(args.question_ids),
        limit=args.limit,
    )
    problems = [problem for problem in problems if problem.question_id in source_records]
    if not problems:
        raise ValueError("no selected problems overlap source generations")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = RUNNER.load_model(
        model_name_or_path=args.model,
        adapter_path=args.adapter_path,
        load_in_4bit=not args.no_4bit,
        bf16=not args.no_bf16,
        attn_implementation=args.attn_implementation,
    )

    generation_records: list[dict[str, Any]] = []
    source_candidate_total = 0
    generated_total = 0
    for index, problem in enumerate(problems, start=1):
        source_record = source_records[problem.question_id]
        source_texts = select_source_texts(
            source_record,
            field=args.source_field,
            max_source_candidates=args.max_source_candidates,
            candidate_indices=candidate_indices,
        )
        if not source_texts:
            continue
        raw_outputs: list[str] = []
        code_outputs: list[str] = []
        used_source_indices: list[int] = []
        print(
            f"[{index}/{len(problems)}] regenerating {problem.question_id} "
            f"{problem.question_title} source_candidates={len(source_texts)}"
        )
        for source_candidate_index, source_text in source_texts:
            prompt = build_second_pass_prompt(
                problem=problem,
                source_text=source_text,
                reasoning_max_chars=args.reasoning_max_chars,
            )
            batch_outputs = RUNNER.generate_one_batch(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                prompt_rendering=args.prompt_rendering,
                system_message=args.system_message,
                response_prefix=args.response_prefix,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                num_return_sequences=args.n_samples_per_source,
                stop_after_code_block=args.stop_after_code_block,
            )
            raw_outputs.extend(batch_outputs)
            code_outputs.extend(RUNNER.strip_lcb_code_block(output) for output in batch_outputs)
            used_source_indices.extend([source_candidate_index] * len(batch_outputs))
            source_candidate_total += 1
            generated_total += len(batch_outputs)
            partial_record = build_generation_record(
                problem=problem,
                prompt=prompt,
                source_generations=str(source_generations_path),
                source_candidate_indices=used_source_indices,
                raw_outputs=raw_outputs,
                code_outputs=code_outputs,
            )
            generations_path.write_text(
                json.dumps(generation_records + [partial_record], indent=2, ensure_ascii=True)
                + "\n",
                encoding="utf-8",
            )
        generation_records.append(
            build_generation_record(
                problem=problem,
                prompt=prompt,
                source_generations=str(source_generations_path),
                source_candidate_indices=used_source_indices,
                raw_outputs=raw_outputs,
                code_outputs=code_outputs,
            )
        )
        generations_path.write_text(
            json.dumps(generation_records, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    report = {
        "benchmark": "LiveCodeBench second-pass final-answer regeneration",
        "lcb_repo": args.lcb_repo,
        "lcb_repo_commit": args.lcb_commit,
        "parquet": args.parquet,
        "parquet_sha256": RUNNER.sha256_file(Path(args.parquet)),
        "source_generations": str(source_generations_path),
        "source_generations_sha256": RUNNER.sha256_file(source_generations_path),
        "output_generations": str(generations_path),
        "model": args.model,
        "adapter_path": args.adapter_path,
        "problems": len(generation_records),
        "source_field": args.source_field,
        "max_source_candidates": args.max_source_candidates,
        "n_samples_per_source": args.n_samples_per_source,
        "source_candidate_total": source_candidate_total,
        "generated_total": generated_total,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "attn_implementation": args.attn_implementation,
        "prompt_rendering": args.prompt_rendering,
        "response_prefix": args.response_prefix,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "reasoning_max_chars": args.reasoning_max_chars,
        "stop_after_code_block": args.stop_after_code_block,
        "seed": args.seed,
        "load_in_4bit": not args.no_4bit,
        "bf16": not args.no_bf16,
        "seconds": round(time.monotonic() - started, 3),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate final LCB answers from saved reasoning-heavy attempts."
    )
    parser.add_argument("--lcb-repo", required=True)
    parser.add_argument("--lcb-commit", default="unknown")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--source-generations", required=True)
    parser.add_argument("--source-field", choices=["raw_outputs", "code_list"], default="raw_outputs")
    parser.add_argument("--source-candidate-indices")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--question-ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-source-candidates", type=int, default=1)
    parser.add_argument("--n-samples-per-source", type=int, default=1)
    parser.add_argument("--reasoning-max-chars", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
    )
    parser.add_argument("--prompt-rendering", choices=["chat", "raw"], default="raw")
    parser.add_argument("--system-message", default=SECOND_PASS_SYSTEM_MESSAGE)
    parser.add_argument("--response-prefix", default="")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--stop-after-code-block", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    regenerate(build_parser().parse_args())


if __name__ == "__main__":
    main()
