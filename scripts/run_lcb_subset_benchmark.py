#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)

FORMAT_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)

FORMAT_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


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


def build_lcb_generation_prompt(question: Any) -> str:
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"### Format: {FORMAT_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMAT_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def strip_lcb_code_block(text: str) -> str:
    stripped = text.strip()
    if "```" not in stripped:
        return stripped

    blocks: list[str] = []
    parts = stripped.split("```")
    for index in range(1, len(parts), 2):
        block = parts[index].strip()
        lines = block.splitlines()
        if lines and lines[0].strip().lower() in {"python", "py"}:
            block = "\n".join(lines[1:]).strip()
        blocks.append(block)

    if not blocks:
        return stripped

    python_blocks = [
        block
        for block in blocks
        if "def " in block or "import " in block or "input(" in block or "sys.stdin" in block
    ]
    return (python_blocks or blocks)[-1].strip()


def parse_iso_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def load_lcb_modules(lcb_repo: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str(lcb_repo))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    return CodeGenerationProblem, codegen_metrics


def load_problems_from_parquet(
    parquet_path: Path,
    lcb_repo: Path,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    difficulty: str | None = None,
    limit: int | None = None,
) -> list[Any]:
    import pyarrow.parquet as pq

    CodeGenerationProblem, _ = load_lcb_modules(lcb_repo)
    rows = pq.read_table(parquet_path).to_pylist()
    problems = [CodeGenerationProblem(**row) for row in rows]
    problems = sorted(problems, key=lambda item: item.question_id)
    if start_date is not None:
        problems = [problem for problem in problems if start_date <= problem.contest_date]
    if end_date is not None:
        problems = [problem for problem in problems if problem.contest_date <= end_date]
    if difficulty is not None:
        problems = [problem for problem in problems if problem.difficulty.value == difficulty]
    if limit is not None:
        problems = problems[:limit]
    return problems


def load_model(
    model_name_or_path: str,
    adapter_path: str | None,
    load_in_4bit: bool,
    bf16: bool,
) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name_or_path,
        trust_remote_code=True,
        local_files_only=Path(model_name_or_path).exists(),
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
    }
    if load_in_4bit:
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_one_batch(
    model: Any,
    tokenizer: Any,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_input_tokens: int,
    num_return_sequences: int,
) -> list[str]:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "num_return_sequences": num_return_sequences,
            }
        )
    else:
        kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **kwargs)

    prompt_tokens = encoded["input_ids"].shape[1]
    return [
        tokenizer.decode(output[prompt_tokens:], skip_special_tokens=True)
        for output in output_ids
    ]


def generate_problem_outputs(
    model: Any,
    tokenizer: Any,
    problem: Any,
    n_samples: int,
    sample_batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_input_tokens: int,
) -> tuple[list[str], list[str]]:
    prompt = build_lcb_generation_prompt(problem)
    raw_outputs: list[str] = []
    code_outputs: list[str] = []
    while len(raw_outputs) < n_samples:
        batch_size = min(sample_batch_size, n_samples - len(raw_outputs))
        if temperature == 0:
            batch_size = 1
        batch_outputs = generate_one_batch(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            num_return_sequences=batch_size,
        )
        raw_outputs.extend(batch_outputs)
        code_outputs.extend(strip_lcb_code_block(output) for output in batch_outputs)
    return raw_outputs[:n_samples], code_outputs[:n_samples]


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    parquet_path = Path(args.parquet)
    output_dir = Path(args.output_dir)
    lcb_repo = Path(args.lcb_repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace files")

    _, codegen_metrics = load_lcb_modules(lcb_repo)
    problems = load_problems_from_parquet(
        parquet_path=parquet_path,
        lcb_repo=lcb_repo,
        start_date=parse_iso_date(args.start_date),
        end_date=parse_iso_date(args.end_date),
        difficulty=args.difficulty,
        limit=args.limit,
    )
    if not problems:
        raise ValueError("no LiveCodeBench problems selected")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    generation_started = time.monotonic()
    model, tokenizer = load_model(
        model_name_or_path=args.model,
        adapter_path=args.adapter_path,
        load_in_4bit=not args.no_4bit,
        bf16=not args.no_bf16,
    )
    generations: list[list[str]] = []
    generation_records: list[dict[str, Any]] = []
    for index, problem in enumerate(problems, start=1):
        print(f"[{index}/{len(problems)}] generating {problem.question_id} {problem.question_title}")
        raw_outputs, code_outputs = generate_problem_outputs(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            n_samples=args.n_samples,
            sample_batch_size=args.sample_batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
        )
        generations.append(code_outputs)
        generation_records.append(
            {
                "question_id": problem.question_id,
                "question_title": problem.question_title,
                "contest_date": problem.contest_date.isoformat(),
                "platform": problem.platform.value,
                "difficulty": problem.difficulty.value,
                "prompt": build_lcb_generation_prompt(problem),
                "raw_outputs": raw_outputs,
                "code_list": code_outputs,
            }
        )
        (output_dir / "generations.json").write_text(
            json.dumps(generation_records, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    generation_seconds = round(time.monotonic() - generation_started, 3)
    eval_started = time.monotonic()
    eval_samples = [problem.get_evaluation_sample() for problem in problems]
    metrics, results, metadata = codegen_metrics(
        eval_samples,
        generations,
        k_list=[1],
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
        debug=args.debug,
    )
    eval_seconds = round(time.monotonic() - eval_started, 3)

    pass_lists = [
        [all(item is True for item in candidate_result) for candidate_result in results[index]]
        for index in range(len(problems))
    ]
    sanitized_metadata = sanitize_lcb_metadata(metadata)
    eval_all = [
        problem.insert_output_evaluation(
            output_list=record["raw_outputs"],
            code_list=record["code_list"],
            graded_list=pass_list,
            metadata=sanitized_metadata[index],
        )
        for index, (problem, record, pass_list) in enumerate(
            zip(problems, generation_records, pass_lists)
        )
    ]

    report = {
        "benchmark": "LiveCodeBench code_generation_lite",
        "benchmark_scope": "release_v6 parquet shard subset; not a full leaderboard run",
        "lcb_repo": str(lcb_repo),
        "lcb_repo_commit": args.lcb_commit,
        "parquet": str(parquet_path),
        "parquet_sha256": sha256_file(parquet_path),
        "parquet_rows_selected": len(problems),
        "model": args.model,
        "adapter_path": args.adapter_path,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "seed": args.seed,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "difficulty": args.difficulty,
        "limit": args.limit,
        "load_in_4bit": not args.no_4bit,
        "bf16": not args.no_bf16,
        "generation_seconds": generation_seconds,
        "evaluation_seconds": eval_seconds,
        "metrics": make_json_safe(metrics),
        "passed_at_1_count": sum(pass_list[0] for pass_list in pass_lists if pass_list),
        "total": len(problems),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(
            make_json_safe([metrics, results, sanitized_metadata]),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
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
    parser = argparse.ArgumentParser(
        description="Run a reproducible LiveCodeBench code-generation-lite subset benchmark."
    )
    parser.add_argument("--lcb-repo", required=True)
    parser.add_argument("--lcb-commit", default=None)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--sample-batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--num-process-evaluate", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = build_parser()
    args = parser.parse_args()
    if args.lcb_commit is None:
        args.lcb_commit = "unknown"
    run_benchmark(args)


if __name__ == "__main__":
    main()
