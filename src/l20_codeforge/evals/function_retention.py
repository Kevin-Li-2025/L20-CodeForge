from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from l20_codeforge.data.code_rlvr import sha256_file
from l20_codeforge.evals.code_rlvr import format_code_prompt, load_jsonl, load_rollout_keys
from l20_codeforge.rewards.code_execution import (
    CodeExecutionConfig,
    _run_limited_process,
    extract_python_code,
    find_python_safety_violation,
)


def generate_function_retention_rollouts(
    model_name_or_path: str,
    tasks_jsonl: Path,
    output: Path,
    *,
    adapter_path: str | None = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_prompt_length: int = 4096,
    max_new_tokens: int = 1024,
    batch_size: int = 2,
    load_in_4bit: bool = True,
    bf16: bool = True,
    seed: int = 42,
    shard_index: int = 0,
    shard_count: int = 1,
    timeout_seconds: float = 4.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate code and execute assertion-style function retention tasks."""

    if n_samples <= 0 or batch_size <= 0:
        raise ValueError("n_samples and batch_size must be positive")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    all_tasks = load_jsonl(tasks_jsonl)
    tasks = [task for index, task in enumerate(all_tasks) if index % shard_count == shard_index]
    if not tasks:
        raise ValueError(f"no tasks selected from shard {shard_index}/{shard_count}")

    torch.manual_seed(seed + shard_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + shard_index)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name_or_path,
        trust_remote_code=True,
        local_files_only=Path(model_name_or_path).exists(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if bf16 else torch.float16
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
        "local_files_only": Path(model_name_or_path).exists(),
    }
    if load_in_4bit:
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    if not load_in_4bit:
        model = model.to("cuda")
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        output.unlink(missing_ok=True)
    existing = load_rollout_keys(output)
    requests = [
        (task, sample_index)
        for task in tasks
        for sample_index in range(n_samples)
        if (str(task["task_id"]), sample_index) not in existing
    ]
    execution_config = CodeExecutionConfig(
        timeout_seconds=timeout_seconds,
        compile_timeout_seconds=timeout_seconds,
        workers=1,
    )
    started = time.monotonic()
    generated = 0
    with output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            prompt_texts = [format_code_prompt(tokenizer, task) for task, _ in batch]
            encoded = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            ).to(model.device)
            generation_kwargs: dict[str, Any] = {
                "do_sample": temperature > 0,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs.update({"temperature": temperature, "top_p": top_p})
            with torch.inference_mode():
                generated_ids = model.generate(**encoded, **generation_kwargs)
            prefix_length = encoded["input_ids"].shape[1]
            completions = tokenizer.batch_decode(
                generated_ids[:, prefix_length:], skip_special_tokens=True
            )
            for (task, sample_index), completion in zip(batch, completions, strict=True):
                code = extract_python_code(completion)
                evaluation = evaluate_function_assertions(
                    code,
                    setup_code=str(task.get("test_setup_code") or ""),
                    tests=[str(test) for test in task["tests"]],
                    config=execution_config,
                )
                row = {
                    "task_id": task["task_id"],
                    "sample_index": sample_index,
                    "prompt_sha256": task.get("prompt_sha256"),
                    "tests_sha256": task.get("tests_sha256"),
                    "prompt": task["prompt"],
                    "completion": completion,
                    "code": code,
                    **evaluation,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                generated += 1

    rows = load_jsonl(output)
    task_ids = {str(row["task_id"]) for row in rows}
    greedy = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id not in greedy or int(row["sample_index"]) < int(greedy[task_id]["sample_index"]):
            greedy[task_id] = row
    greedy_passes = sum(bool(row["all_passed"]) for row in greedy.values())
    report = {
        "model_name_or_path": model_name_or_path,
        "adapter_path": adapter_path,
        "tasks_jsonl": str(tasks_jsonl),
        "tasks_jsonl_sha256": sha256_file(tasks_jsonl),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "tasks": len(task_ids),
        "expected_tasks": len(tasks),
        "samples": len(rows),
        "greedy_passes": greedy_passes,
        "greedy_accuracy": greedy_passes / len(tasks),
        "generated_this_run": generated,
        "n_samples": n_samples,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "claim_boundary": (
            "Retention-development accuracy on official MBPP validation examples only; "
            "not MBPP test or MBPP+ evidence."
        ),
    }
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def evaluate_function_assertions(
    code: str,
    *,
    setup_code: str,
    tests: list[str],
    config: CodeExecutionConfig | None = None,
) -> dict[str, Any]:
    """Compile and execute public assertion tests in the bounded subprocess runner."""

    if not tests:
        raise ValueError("function retention tasks require at least one assertion")
    active_config = config or CodeExecutionConfig(timeout_seconds=4.0)
    violation = find_python_safety_violation(code)
    if violation:
        return {
            "status": "unsafe",
            "compiled": False,
            "all_passed": False,
            "exit_code": 126,
            "stderr": f"unsafe code rejected: {violation}",
            "elapsed_seconds": 0.0,
        }
    source = "\n\n".join(
        part for part in [code.rstrip(), setup_code.strip(), "\n".join(tests)] if part
    ) + "\n"
    with tempfile.TemporaryDirectory(prefix="l20-function-retention-") as temp_dir:
        workdir = Path(temp_dir)
        solution = workdir / "solution.py"
        solution.write_text(source, encoding="utf-8")
        compile_result = _run_limited_process(
            [sys.executable, "-I", "-m", "py_compile", str(solution)],
            cwd=workdir,
            stdin="",
            config=active_config,
            timeout_seconds=active_config.compile_timeout_seconds,
        )
        if compile_result.exit_code != 0:
            return {
                "status": "compile_error",
                "compiled": False,
                "all_passed": False,
                "exit_code": compile_result.exit_code,
                "stderr": compile_result.stderr,
                "elapsed_seconds": compile_result.elapsed_seconds,
            }
        result = _run_limited_process(
            [sys.executable, "-I", str(solution)],
            cwd=workdir,
            stdin="",
            config=active_config,
        )
    passed = result.exit_code == 0 and not result.timed_out and not result.output_limited
    status = "passed" if passed else "timeout" if result.timed_out else "failed"
    return {
        "status": status,
        "compiled": True,
        "all_passed": passed,
        "exit_code": result.exit_code,
        "stderr": result.stderr,
        "elapsed_seconds": result.elapsed_seconds,
    }
