#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


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
    tie_breaker: str = "shortest",
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


def build_lcb_generation_prompt(question: Any, prompt_suffix: str = "") -> str:
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"### Format: {FORMAT_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMAT_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    if prompt_suffix:
        prompt += prompt_suffix.strip() + "\n\n"
    return prompt


def strip_lcb_code_block(text: str) -> str:
    stripped = text.strip()
    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_match:
        return strip_lcb_code_block(answer_match.group(1))
    if "</think>" in stripped.lower():
        after_think = re.split(
            r"</think>",
            stripped,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[1].strip()
        if after_think:
            return strip_lcb_code_block(after_think)
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


def parse_question_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    question_ids = {item.strip() for item in value.split(",") if item.strip()}
    if not question_ids:
        raise ValueError("--question-ids must contain at least one id")
    return question_ids


def load_lcb_modules(lcb_repo: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str(lcb_repo))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    return CodeGenerationProblem, codegen_metrics


def load_generation_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list")
    return records


def load_resume_generation_records(
    generations_path: Path,
    resume_from_generations: Path | None,
) -> tuple[list[dict[str, Any]], str | None]:
    seed_records = (
        load_generation_records(resume_from_generations)
        if resume_from_generations is not None
        else []
    )
    output_records = (
        load_generation_records(generations_path) if generations_path.exists() else []
    )
    if seed_records and output_records:
        merged_by_id = {
            record["question_id"]: record
            for record in seed_records
            if isinstance(record.get("question_id"), str)
        }
        merged_order = [
            record["question_id"]
            for record in seed_records
            if isinstance(record.get("question_id"), str)
        ]
        for record in output_records:
            question_id = record.get("question_id")
            if not isinstance(question_id, str):
                continue
            if question_id not in merged_by_id:
                merged_order.append(question_id)
            merged_by_id[question_id] = record
        return [merged_by_id[question_id] for question_id in merged_order], (
            f"{generations_path} over {resume_from_generations}"
        )
    if output_records:
        return output_records, str(generations_path)
    if seed_records:
        return seed_records, str(resume_from_generations)
    return [], None


def validate_resume_records(
    records: list[dict[str, Any]],
    n_samples: int,
    allow_partial: bool = False,
) -> dict[str, dict[str, Any]]:
    resumed: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = record.get("question_id")
        code_list = record.get("code_list") or []
        raw_outputs = record.get("raw_outputs") or []
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("resume generation record is missing question_id")
        if question_id in resumed:
            raise ValueError(f"duplicate resume generation record for {question_id}")
        available_samples = min(len(code_list), len(raw_outputs))
        if available_samples < n_samples and not allow_partial:
            raise ValueError(
                f"resume generation record for {question_id} has fewer than "
                f"{n_samples} samples"
            )
        record["code_list"] = code_list[: min(available_samples, n_samples)]
        record["raw_outputs"] = raw_outputs[: min(available_samples, n_samples)]
        resumed[question_id] = record
    return resumed


def load_problems_from_parquet(
    parquet_path: Path,
    lcb_repo: Path,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    difficulty: str | None = None,
    question_ids: set[str] | None = None,
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
    if question_ids is not None:
        problems = [problem for problem in problems if problem.question_id in question_ids]
    if limit is not None:
        problems = problems[:limit]
    return problems


def build_base_model_kwargs(
    torch_module: Any,
    load_in_4bit: bool,
    bf16: bool,
    attn_implementation: str,
) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_module.bfloat16 if bf16 else torch_module.float16,
    }
    if attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit or torch_module.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    return model_kwargs


def load_model(
    model_name_or_path: str,
    adapter_path: str | None,
    load_in_4bit: bool,
    bf16: bool,
    attn_implementation: str,
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

    model_kwargs = build_base_model_kwargs(
        torch_module=torch,
        load_in_4bit=load_in_4bit,
        bf16=bf16,
        attn_implementation=attn_implementation,
    )
    if load_in_4bit:
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


def build_generation_record(
    problem: Any,
    prompt_suffix: str,
    raw_outputs: list[str],
    code_outputs: list[str],
) -> dict[str, Any]:
    return {
        "question_id": problem.question_id,
        "question_title": problem.question_title,
        "contest_date": problem.contest_date.isoformat(),
        "platform": problem.platform.value,
        "difficulty": problem.difficulty.value,
        "prompt": build_lcb_generation_prompt(
            problem,
            prompt_suffix=prompt_suffix,
        ),
        "raw_outputs": raw_outputs,
        "code_list": code_outputs,
    }


def render_model_input(
    tokenizer: Any,
    prompt: str,
    prompt_rendering: str,
    system_message: str,
) -> str:
    if prompt_rendering == "raw":
        return prompt
    if prompt_rendering == "chat":
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError("prompt_rendering must be one of: chat, raw")


def generated_text_has_closed_code_block(text: str) -> bool:
    return text.count("```") >= 2


def generate_one_batch(
    model: Any,
    tokenizer: Any,
    prompt: str,
    prompt_rendering: str,
    system_message: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_new_tokens: int,
    max_input_tokens: int,
    num_return_sequences: int,
    stop_after_code_block: bool,
) -> list[str]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    rendered = render_model_input(
        tokenizer=tokenizer,
        prompt=prompt,
        prompt_rendering=prompt_rendering,
        system_message=system_message,
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
        if top_k is not None and top_k >= 0:
            kwargs["top_k"] = top_k
    else:
        kwargs["do_sample"] = False

    if stop_after_code_block:
        prompt_tokens_for_stop = encoded["input_ids"].shape[1]

        class StopAfterClosedCodeBlock(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                del scores, kwargs
                return all(
                    generated_text_has_closed_code_block(
                        tokenizer.decode(
                            output[prompt_tokens_for_stop:],
                            skip_special_tokens=True,
                        )
                    )
                    for output in input_ids
                )

        kwargs["stopping_criteria"] = StoppingCriteriaList([StopAfterClosedCodeBlock()])

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
    prompt_suffix: str,
    prompt_rendering: str,
    system_message: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_new_tokens: int,
    max_input_tokens: int,
    stop_after_code_block: bool,
    progress_callback: Callable[[list[str], list[str]], None] | None = None,
) -> tuple[list[str], list[str]]:
    prompt = build_lcb_generation_prompt(problem, prompt_suffix=prompt_suffix)
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
            prompt_rendering=prompt_rendering,
            system_message=system_message,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            num_return_sequences=batch_size,
            stop_after_code_block=stop_after_code_block,
        )
        raw_outputs.extend(batch_outputs)
        code_outputs.extend(strip_lcb_code_block(output) for output in batch_outputs)
        if progress_callback is not None:
            progress_callback(raw_outputs[:n_samples], code_outputs[:n_samples])
    return raw_outputs[:n_samples], code_outputs[:n_samples]


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    parquet_path = Path(args.parquet)
    output_dir = Path(args.output_dir)
    lcb_repo = Path(args.lcb_repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / "generations.json"
    if any(output_dir.iterdir()) and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"{output_dir} is not empty; pass --overwrite or --resume to continue"
        )
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    _, codegen_metrics = load_lcb_modules(lcb_repo)
    problems = load_problems_from_parquet(
        parquet_path=parquet_path,
        lcb_repo=lcb_repo,
        start_date=parse_iso_date(args.start_date),
        end_date=parse_iso_date(args.end_date),
        difficulty=args.difficulty,
        question_ids=parse_question_ids(args.question_ids),
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
        attn_implementation=args.attn_implementation,
    )
    resume_from_generations = (
        Path(args.resume_from_generations) if args.resume_from_generations else None
    )
    existing_records, resume_generations_source = (
        load_resume_generation_records(generations_path, resume_from_generations)
        if args.resume
        else ([], None)
    )
    resumed_records = validate_resume_records(
        existing_records,
        args.n_samples,
        allow_partial=args.allow_partial_resume,
    )
    generation_records: list[dict[str, Any]] = []
    generated_this_run = 0
    samples_generated_this_run = 0
    for index, problem in enumerate(problems, start=1):
        existing_record = resumed_records.get(problem.question_id)
        existing_sample_count = (
            len(existing_record.get("code_list", [])) if existing_record else 0
        )
        if existing_record is not None and existing_sample_count >= args.n_samples:
            print(
                f"[{index}/{len(problems)}] resuming {problem.question_id} "
                f"{problem.question_title}"
            )
            generation_records.append(existing_record)
            continue
        existing_raw_outputs = (
            list(existing_record.get("raw_outputs", [])) if existing_record else []
        )
        existing_code_outputs = (
            list(existing_record.get("code_list", [])) if existing_record else []
        )
        missing_samples = args.n_samples - len(existing_code_outputs)
        action = "extending" if existing_record is not None else "generating"
        print(
            f"[{index}/{len(problems)}] {action} {problem.question_id} "
            f"{problem.question_title} missing_samples={missing_samples}"
        )

        def persist_partial_progress(
            partial_raw_outputs: list[str],
            partial_code_outputs: list[str],
        ) -> None:
            partial_record = build_generation_record(
                problem=problem,
                prompt_suffix=args.prompt_suffix,
                raw_outputs=existing_raw_outputs + partial_raw_outputs,
                code_outputs=existing_code_outputs + partial_code_outputs,
            )
            generations_path.write_text(
                json.dumps(
                    generation_records + [partial_record],
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

        raw_outputs, code_outputs = generate_problem_outputs(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            n_samples=missing_samples,
            sample_batch_size=args.sample_batch_size,
            prompt_suffix=args.prompt_suffix,
            prompt_rendering=args.prompt_rendering,
            system_message=args.system_message,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            stop_after_code_block=args.stop_after_code_block,
            progress_callback=persist_partial_progress,
        )
        raw_outputs = existing_raw_outputs + raw_outputs
        code_outputs = existing_code_outputs + code_outputs
        generation_records.append(
            build_generation_record(
                problem=problem,
                prompt_suffix=args.prompt_suffix,
                raw_outputs=raw_outputs,
                code_outputs=code_outputs,
            )
        )
        generated_this_run += 1
        samples_generated_this_run += missing_samples
        generations_path.write_text(
            json.dumps(generation_records, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    generations = [record["code_list"] for record in generation_records]
    generations_path.write_text(
        json.dumps(generation_records, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    generation_seconds = round(time.monotonic() - generation_started, 3)
    if args.generate_only:
        report = {
            "benchmark": "LiveCodeBench code_generation_lite",
            "benchmark_scope": "generation only; evaluation skipped",
            "lcb_repo": str(lcb_repo),
            "lcb_repo_commit": args.lcb_commit,
            "parquet": str(parquet_path),
            "parquet_sha256": sha256_file(parquet_path),
            "parquet_rows_selected": len(problems),
            "model": args.model,
            "adapter_path": args.adapter_path,
            "n_samples": args.n_samples,
            "selection": "none",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "attn_implementation": args.attn_implementation,
            "prompt_rendering": args.prompt_rendering,
            "prompt_suffix": args.prompt_suffix,
            "system_message": args.system_message if args.prompt_rendering == "chat" else None,
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "stop_after_code_block": args.stop_after_code_block,
            "seed": args.seed,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "difficulty": args.difficulty,
            "question_ids": args.question_ids,
            "limit": args.limit,
            "load_in_4bit": not args.no_4bit,
            "bf16": not args.no_bf16,
            "generation_seconds": generation_seconds,
            "resume": args.resume,
            "allow_partial_resume": args.allow_partial_resume,
            "resume_generations_source": resume_generations_source,
            "resumed_count": len(existing_records),
            "generated_this_run": generated_this_run,
            "samples_generated_this_run": samples_generated_this_run,
            "evaluation_seconds": 0.0,
            "metrics": None,
            "passed_at_1_count": None,
            "total": len(problems),
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return report

    public_selection_seconds = 0.0
    public_selection_records: list[dict[str, Any]] = []
    public_selection_metrics: dict[str, Any] | None = None
    public_selection_metadata: Any | None = None
    final_generations = generations
    final_raw_outputs = [record["raw_outputs"] for record in generation_records]
    if args.public_select:
        public_selection_started = time.monotonic()
        public_samples = [get_public_evaluation_sample(problem) for problem in problems]
        public_k_list = sorted({1, args.n_samples})
        public_metrics, public_results, public_metadata = codegen_metrics(
            public_samples,
            generations,
            k_list=public_k_list,
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.public_select_timeout,
            debug=args.debug,
        )
        final_generations, public_selection_records = build_public_selection_records(
            problems=problems,
            generations=generations,
            public_results=public_results,
            tie_breaker=args.public_select_tie_breaker,
        )
        final_raw_outputs = [
            [generation_records[index]["raw_outputs"][record["selected_index"]]]
            for index, record in enumerate(public_selection_records)
        ]
        public_selection_seconds = round(time.monotonic() - public_selection_started, 3)
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

    eval_started = time.monotonic()
    eval_samples = [problem.get_evaluation_sample() for problem in problems]
    metrics, results, metadata = codegen_metrics(
        eval_samples,
        final_generations,
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
            output_list=final_raw_outputs[index],
            code_list=final_generations[index],
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
        "selection": "public_tests" if args.public_select else "none",
        "public_select_tie_breaker": args.public_select_tie_breaker
        if args.public_select
        else None,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "attn_implementation": args.attn_implementation,
        "prompt_rendering": args.prompt_rendering,
        "prompt_suffix": args.prompt_suffix,
        "system_message": args.system_message if args.prompt_rendering == "chat" else None,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "stop_after_code_block": args.stop_after_code_block,
        "seed": args.seed,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "difficulty": args.difficulty,
        "question_ids": args.question_ids,
        "limit": args.limit,
        "load_in_4bit": not args.no_4bit,
        "bf16": not args.no_bf16,
        "generation_seconds": generation_seconds,
        "resume": args.resume,
        "allow_partial_resume": args.allow_partial_resume,
        "resume_generations_source": resume_generations_source,
        "resumed_count": len(existing_records),
        "generated_this_run": generated_this_run,
        "samples_generated_this_run": samples_generated_this_run,
        "public_selection_seconds": public_selection_seconds,
        "evaluation_seconds": eval_seconds,
        "metrics": make_json_safe(metrics),
        "public_selection_metrics": public_selection_metrics,
        "public_selected_pass_count": sum(
            record["selected_public_score"] == 1.0 for record in public_selection_records
        )
        if args.public_select
        else None,
        "public_oracle_pass_count": sum(
            record["public_oracle_pass"] for record in public_selection_records
        )
        if args.public_select
        else None,
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
    parser.add_argument(
        "--question-ids",
        help="Optional comma-separated LiveCodeBench question IDs to select.",
    )
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--sample-batch-size", type=int, default=1)
    parser.add_argument("--public-select", action="store_true")
    parser.add_argument(
        "--public-select-tie-breaker",
        choices=["first", "shortest", "longest"],
        default="shortest",
    )
    parser.add_argument("--public-select-timeout", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
        help="Optional transformers attention backend override.",
    )
    parser.add_argument(
        "--prompt-rendering",
        choices=["chat", "raw"],
        default="chat",
        help="Use chat template wrapping or feed the benchmark prompt as raw text.",
    )
    parser.add_argument(
        "--prompt-suffix",
        default="",
        help="Optional text appended after the benchmark answer instruction.",
    )
    parser.add_argument("--system-message", default=SYSTEM_MESSAGE_GENERIC)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument(
        "--stop-after-code-block",
        action="store_true",
        help="Stop generation after every returned sequence has a closed fenced code block.",
    )
    parser.add_argument("--num-process-evaluate", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-from-generations",
        help=(
            "Optional source generations.json used to seed a new output directory "
            "when --resume is set and the output generations file does not exist."
        ),
    )
    parser.add_argument(
        "--allow-partial-resume",
        action="store_true",
        help=(
            "When --resume is set, allow existing generation records with fewer "
            "than --n-samples and generate only the missing samples. This is used "
            "to extend an n=4 run to n=8/n=16 without regenerating prior samples."
        ),
    )
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
