from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvalPlusGenerationReport(BaseModel):
    dataset: str
    model_name_or_path: str
    adapter_path: str | None = None
    output: str
    raw_output: str
    tasks_seen: int
    samples_written: int
    n_samples: int
    limit: int | None = None
    id_start: int | None = None
    id_end: int | None = None
    temperature: float = 0.0
    top_p: float = 0.95
    max_new_tokens: int = 512
    sample_batch_size: int = 1
    load_in_4bit: bool = True
    elapsed_seconds: float = 0.0


class EvalPlusOfficialReport(BaseModel):
    dataset: str
    samples: str
    exit_code: int
    base_pass_at_1: float | None = None
    plus_pass_at_1: float | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    result_files: list[str] = Field(default_factory=list)


class EvalPlusSelectionReport(BaseModel):
    samples: str
    eval_results: str
    output: str
    tasks: int
    selected_base_pass: int
    selected_plus_pass: int | None = None
    fallback_tasks: list[str] = Field(default_factory=list)


def generate_evalplus_samples(
    model_name_or_path: str,
    dataset: str,
    output: Path,
    adapter_path: str | None = None,
    n_samples: int = 1,
    limit: int | None = None,
    id_start: int | None = None,
    id_end: int | None = None,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    sample_batch_size: int = 1,
    load_in_4bit: bool = True,
    bf16: bool = True,
    seed: int = 42,
    overwrite: bool = False,
) -> EvalPlusGenerationReport:
    import torch
    from evalplus.sanitize import sanitize
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tasks = select_evalplus_tasks(
        load_evalplus_tasks(dataset),
        limit=limit,
        id_start=id_start,
        id_end=id_end,
    )
    if not tasks:
        raise ValueError(f"no EvalPlus tasks selected for dataset={dataset!r}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name_or_path,
        trust_remote_code=True,
        local_files_only=Path(model_name_or_path).exists(),
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

    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_suffix(".raw.jsonl")
    if overwrite:
        output.unlink(missing_ok=True)
        raw_output.unlink(missing_ok=True)

    existing = count_existing_samples(output)
    started = time.monotonic()
    samples_written = 0
    with output.open("a", encoding="utf-8") as sanitized_handle, raw_output.open(
        "a", encoding="utf-8"
    ) as raw_handle:
        for task_id, task in tasks:
            already = existing.get(task_id, 0)
            if already >= n_samples:
                continue
            sample_index = already
            while sample_index < n_samples:
                batch_size = min(sample_batch_size, n_samples - sample_index)
                if temperature == 0:
                    batch_size = 1
                prompt = build_evalplus_prompt(task["prompt"])
                completions = generate_many(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=batch_size,
                )
                for completion in completions:
                    code_text = strip_markdown_code_fence(completion)
                    solution_input = task["prompt"].rstrip() + "\n" + code_text
                    sanitized = sanitize(solution_input, entrypoint=task["entry_point"])
                    if not sanitized.strip():
                        sanitized = solution_input

                    sanitized_handle.write(
                        json.dumps({"task_id": task_id, "solution": sanitized}) + "\n"
                    )
                    raw_handle.write(
                        json.dumps(
                            {
                                "task_id": task_id,
                                "sample_index": sample_index,
                                "prompt": prompt,
                                "completion": completion,
                                "solution_input": solution_input,
                                "sanitized": sanitized,
                            }
                        )
                        + "\n"
                    )
                    sanitized_handle.flush()
                    raw_handle.flush()
                    samples_written += 1
                    sample_index += 1

    return EvalPlusGenerationReport(
        dataset=dataset,
        model_name_or_path=model_name_or_path,
        adapter_path=adapter_path,
        output=str(output),
        raw_output=str(raw_output),
        tasks_seen=len(tasks),
        samples_written=samples_written,
        n_samples=n_samples,
        limit=limit,
        id_start=id_start,
        id_end=id_end,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        sample_batch_size=sample_batch_size,
        load_in_4bit=load_in_4bit,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )


def run_evalplus_official(
    dataset: str,
    samples: Path,
    output: Path,
    base_only: bool = False,
    parallel: int | None = None,
    test_details: bool = True,
    mini: bool = False,
    i_just_wanna_run: bool = False,
) -> EvalPlusOfficialReport:
    command = ["evalplus.evaluate", dataset, "--samples", str(samples)]
    if base_only:
        command.append("--base-only")
    if parallel is not None:
        command.extend(["--parallel", str(parallel)])
    if test_details:
        command.append("--test-details")
    if mini:
        command.append("--mini")
    if i_just_wanna_run:
        command.append("--i_just_wanna_run")

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    result_files = []
    expected_result = samples.with_name(samples.stem + "_eval_results.json")
    if expected_result.exists():
        result_files.append(str(expected_result))
    result_files.extend(
        str(path)
        for path in sorted(samples.parent.glob(samples.stem + "*eval_results*.json"))
        if str(path) not in result_files
    )
    parsed_scores = parse_evalplus_scores(completed.stdout)
    report = EvalPlusOfficialReport(
        dataset=dataset,
        samples=str(samples),
        exit_code=completed.returncode,
        base_pass_at_1=parsed_scores.get("base_pass@1"),
        plus_pass_at_1=parsed_scores.get("plus_pass@1"),
        scores=parsed_scores,
        stdout=completed.stdout[-20000:],
        stderr=completed.stderr[-20000:],
        result_files=result_files,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def select_evalplus_by_base_tests(
    samples: Path,
    eval_results: Path,
    output: Path,
) -> EvalPlusSelectionReport:
    grouped_samples = load_evalplus_samples_by_task(samples)
    results = json.loads(eval_results.read_text(encoding="utf-8"))["eval"]
    output.parent.mkdir(parents=True, exist_ok=True)

    selected_base_pass = 0
    selected_plus_pass = 0
    fallback_tasks: list[str] = []
    with output.open("w", encoding="utf-8") as handle:
        for task_id, candidates in grouped_samples.items():
            task_results = results.get(task_id, [])
            selected_index = 0
            for index, result in enumerate(task_results):
                if result.get("base_status") == "pass":
                    selected_index = index
                    break
            else:
                fallback_tasks.append(task_id)

            chosen = candidates[min(selected_index, len(candidates) - 1)]
            handle.write(json.dumps(chosen) + "\n")
            if selected_index < len(task_results):
                selected_result = task_results[selected_index]
                if selected_result.get("base_status") == "pass":
                    selected_base_pass += 1
                if selected_result.get("plus_status") == "pass":
                    selected_plus_pass += 1

    return EvalPlusSelectionReport(
        samples=str(samples),
        eval_results=str(eval_results),
        output=str(output),
        tasks=len(grouped_samples),
        selected_base_pass=selected_base_pass,
        selected_plus_pass=selected_plus_pass,
        fallback_tasks=fallback_tasks,
    )


def load_evalplus_samples_by_task(samples: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    with samples.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            task_id = payload["task_id"]
            grouped.setdefault(task_id, []).append(payload)
    return grouped


def parse_evalplus_pass_at_1(stdout: str) -> dict[str, float]:
    scores = parse_evalplus_scores(stdout)
    return {
        key.removesuffix("_pass@1"): value
        for key, value in scores.items()
        if key.endswith("_pass@1")
    }


def parse_evalplus_scores(stdout: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    current: str | None = None
    lines = [line.strip() for line in stdout.splitlines()]
    for index, line in enumerate(lines):
        if line.endswith("(base tests)"):
            current = "base"
            continue
        if line.endswith("(base + extra tests)"):
            current = "plus"
            continue
        if line.startswith("pass@") and current:
            metric = line.split(":", 1)[0]
            score_text = line.split(":", 1)[1].strip()
            if not score_text and index + 1 < len(lines):
                score_text = lines[index + 1]
            try:
                scores[f"{current}_{metric}"] = float(score_text)
            except ValueError:
                pass
    return scores


def load_evalplus_tasks(dataset: str) -> dict[str, dict[str, Any]]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError("dataset must be 'humaneval' or 'mbpp'")


def select_evalplus_tasks(
    tasks: dict[str, dict[str, Any]],
    limit: int | None = None,
    id_start: int | None = None,
    id_end: int | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    for task_id, task in tasks.items():
        task_num = int(task_id.split("/")[1])
        if id_start is not None and task_num < id_start:
            continue
        if id_end is not None and task_num >= id_end:
            continue
        selected.append((task_id, task))
        if limit is not None and len(selected) >= limit:
            break
    return selected


def count_existing_samples(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            task_id = payload.get("task_id")
            if isinstance(task_id, str):
                counts[task_id] = counts.get(task_id, 0) + 1
    return counts


def build_evalplus_prompt(function_prompt: str) -> str:
    return (
        "Complete the following Python function for an execution-based coding benchmark.\n"
        "Return only valid Python code. Do not include markdown or explanations.\n\n"
        f"{function_prompt.rstrip()}\n"
    )


def generate_many(
    model: Any,
    tokenizer: Any,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    num_return_sequences: int = 1,
) -> list[str]:
    import torch

    chat = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
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
    return [
        tokenizer.decode(
            output[encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        for output in output_ids
    ]


def strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    parts = stripped.split("```")
    for part in parts:
        candidate = part.strip()
        if candidate.startswith("python"):
            return candidate[len("python") :].strip()
    return parts[1].strip() if len(parts) > 1 else stripped
