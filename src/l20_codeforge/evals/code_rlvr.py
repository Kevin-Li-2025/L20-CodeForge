from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from l20_codeforge.data.code_rlvr import DEFAULT_SYSTEM_PROMPT, sha256_file, write_jsonl
from l20_codeforge.rewards.code_execution import (
    CodeExecutionConfig,
    evaluate_python_completion,
    extract_python_code,
)


def generate_code_rollouts(
    model_name_or_path: str,
    tasks_jsonl: Path,
    output: Path,
    *,
    adapter_path: str | None = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_prompt_length: int = 6144,
    max_new_tokens: int = 1536,
    batch_size: int = 2,
    load_in_4bit: bool = True,
    bf16: bool = True,
    seed: int = 42,
    shard_index: int = 0,
    shard_count: int = 1,
    timeout_seconds: float = 2.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate and execute standalone-code completions for one deterministic shard."""

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
    execution_config = CodeExecutionConfig(timeout_seconds=timeout_seconds)
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
            do_sample = temperature > 0
            generation_kwargs: dict[str, Any] = {
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs.update({"temperature": temperature, "top_p": top_p})
            with torch.inference_mode():
                generated_ids = model.generate(**encoded, **generation_kwargs)
            prefix_length = encoded["input_ids"].shape[1]
            completions = tokenizer.batch_decode(
                generated_ids[:, prefix_length:],
                skip_special_tokens=True,
            )
            for (task, sample_index), prompt_text, completion in zip(
                batch, prompt_texts, completions, strict=True
            ):
                report = evaluate_python_completion(
                    completion,
                    task["tests"],
                    execution_config,
                )
                row = {
                    "task_id": task["task_id"],
                    "sample_index": sample_index,
                    "prompt_sha256": task.get("prompt_sha256"),
                    "tests_sha256": task.get("tests_sha256"),
                    "prompt": task["prompt"],
                    "formatted_prompt_sha256": hashlib.sha256(
                        prompt_text.encode("utf-8")
                    ).hexdigest(),
                    "completion": completion,
                    "code": report.extracted_code,
                    "status": report.status,
                    "compiled": report.compiled,
                    "all_passed": report.all_passed,
                    "tests_passed": report.tests_passed,
                    "tests_total": report.tests_total,
                    "pass_fraction": report.pass_fraction,
                    "reward": report.reward,
                    "behavior_signature": report.behavior_signature,
                    "elapsed_seconds": report.elapsed_seconds,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                generated += 1

    summary = summarize_code_rollouts(load_jsonl(output), expected_tasks=len(tasks))
    manifest = {
        "model_name_or_path": model_name_or_path,
        "adapter_path": adapter_path,
        "tasks_jsonl": str(tasks_jsonl),
        "tasks_jsonl_sha256": sha256_file(tasks_jsonl),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "n_samples": n_samples,
        "temperature": temperature,
        "top_p": top_p,
        "max_prompt_length": max_prompt_length,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "load_in_4bit": load_in_4bit,
        "bf16": bf16,
        "seed": seed,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "timeout_seconds": timeout_seconds,
        "generated_this_run": generated,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "summary": summary,
        "claim_boundary": (
            "Executable accuracy on the frozen rStar-derived split only; not EvalPlus "
            "or LiveCodeBench evidence."
        ),
    }
    manifest_path = output.with_suffix(".report.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def format_code_prompt(tokenizer: Any, task: dict[str, Any]) -> str:
    messages = task.get("messages") or [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def merge_code_rollouts(
    inputs: Sequence[Path], output: Path, *, expected_tasks: int | None = None
) -> dict[str, Any]:
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for path in inputs:
        for row in load_jsonl(path):
            key = (str(row["task_id"]), int(row["sample_index"]))
            if key in merged and merged[key] != row:
                raise ValueError(f"conflicting duplicate rollout {key}")
            merged[key] = row
    rows = [merged[key] for key in sorted(merged)]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    report = {
        "inputs": [str(path) for path in inputs],
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "output": str(output),
        "output_sha256": sha256_file(output),
        "summary": summarize_code_rollouts(rows, expected_tasks=expected_tasks),
    }
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def summarize_code_rollouts(
    rows: Iterable[dict[str, Any]], *, expected_tasks: int | None = None
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    task_passes = sum(
        any(bool(row.get("all_passed")) for row in group) for group in grouped.values()
    )
    greedy_rows = [
        min(group, key=lambda row: int(row.get("sample_index", 0))) for group in grouped.values()
    ]
    greedy_passes = sum(bool(row.get("all_passed")) for row in greedy_rows)
    mixed_tasks = sum(
        len({bool(row.get("all_passed")) for row in group}) > 1 for group in grouped.values()
    )
    task_count = len(grouped)
    denominator = expected_tasks or task_count
    return {
        "tasks": task_count,
        "expected_tasks": expected_tasks,
        "missing_tasks": max(0, denominator - task_count),
        "samples": sum(len(group) for group in grouped.values()),
        "greedy_passes": greedy_passes,
        "greedy_accuracy": greedy_passes / denominator if denominator else 0.0,
        "pass_at_n_tasks": task_passes,
        "pass_at_n": task_passes / denominator if denominator else 0.0,
        "mixed_reward_tasks": mixed_tasks,
        "mixed_reward_fraction": mixed_tasks / task_count if task_count else 0.0,
        "mean_pass_fraction": (
            sum(float(row.get("pass_fraction", 0.0)) for group in grouped.values() for row in group)
            / sum(len(group) for group in grouped.values())
            if grouped
            else 0.0
        ),
    }


def build_verified_sft_from_rollouts(
    rollouts_jsonl: Path,
    output: Path,
    *,
    min_distinct_passing: int = 1,
    max_records: int | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(rollouts_jsonl):
        if row.get("all_passed"):
            grouped[str(row["task_id"])].append(row)

    records: list[dict[str, Any]] = []
    skipped_insufficient = 0
    for task_id in sorted(grouped):
        unique_by_code: dict[str, dict[str, Any]] = {}
        for row in grouped[task_id]:
            code = extract_python_code(str(row.get("code") or row.get("completion") or ""))
            code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
            unique_by_code.setdefault(code_hash, {**row, "code": code})
        if len(unique_by_code) < min_distinct_passing:
            skipped_insufficient += 1
            continue
        selected = min(unique_by_code.values(), key=lambda row: len(str(row["code"])))
        records.append(
            {
                "dataset": "base-self-rollouts-execution-verified",
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": selected["prompt"]},
                    {"role": "assistant", "content": str(selected["code"]).rstrip() + "\n"},
                ],
                "metadata": {
                    "task_id": task_id,
                    "prompt_sha256": selected.get("prompt_sha256"),
                    "tests_sha256": selected.get("tests_sha256"),
                    "verification": "all_frozen_tests_passed",
                    "distinct_passing_candidates": len(unique_by_code),
                },
            }
        )
        if max_records is not None and len(records) >= max_records:
            break

    if not records:
        raise ValueError("no execution-verified SFT records were admitted")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, records)
    report = {
        "rollouts_jsonl": str(rollouts_jsonl),
        "rollouts_sha256": sha256_file(rollouts_jsonl),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "records": len(records),
        "min_distinct_passing": min_distinct_passing,
        "skipped_insufficient_distinct_passing": skipped_insufficient,
        "claim_boundary": "Every admitted assistant program passed its frozen executable tests.",
    }
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def select_mixed_reward_tasks(
    tasks_jsonl: Path,
    rollouts_jsonl: Path,
    output: Path,
    *,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    outcomes: dict[str, set[bool]] = defaultdict(set)
    for row in load_jsonl(rollouts_jsonl):
        outcomes[str(row["task_id"])].add(bool(row.get("all_passed")))
    selected = [
        task
        for task in load_jsonl(tasks_jsonl)
        if len(outcomes.get(str(task["task_id"]), set())) > 1
    ]
    if max_tasks is not None:
        selected = selected[:max_tasks]
    if not selected:
        raise ValueError("no tasks have mixed pass/fail rollout rewards")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    report = {
        "tasks_jsonl": str(tasks_jsonl),
        "tasks_sha256": sha256_file(tasks_jsonl),
        "rollouts_jsonl": str(rollouts_jsonl),
        "rollouts_sha256": sha256_file(rollouts_jsonl),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "selected_tasks": len(selected),
        "selection": "at least one pass and one failure among matched base rollouts",
    }
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rollout_keys(path: Path) -> set[tuple[str, int]]:
    return {(str(row["task_id"]), int(row["sample_index"])) for row in load_jsonl(path)}
