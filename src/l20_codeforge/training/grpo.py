from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

from l20_codeforge.data.code_rlvr import sha256_file
from l20_codeforge.training.reward_functions import (
    code_binary_execution_reward,
    code_execution_reward,
)
from l20_codeforge.training.sft import DEFAULT_LORA_TARGETS


def train_code_grpo(
    model_name_or_path: str,
    train_jsonl: Path,
    output_dir: Path,
    *,
    adapter_path: str | None = None,
    max_steps: int = 100,
    max_completion_length: int = 1024,
    limit: int | None = None,
    learning_rate: float = 1e-6,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    num_generations: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.95,
    beta: float = 0.0,
    loss_type: str = "dr_grpo",
    reward_type: str = "dense",
    timeout_seconds: float = 2.0,
    execution_workers: int = 6,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    load_in_4bit: bool = True,
    bf16: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Run executable-reward GRPO, optionally warm-starting from an SFT adapter."""

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    if reward_type not in {"dense", "binary"}:
        raise ValueError("reward_type must be 'dense' or 'binary'")
    rows = _load_grpo_rows(train_jsonl, limit=limit)
    dataset = Dataset.from_list(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if load_in_4bit:
        model_kwargs["device_map"] = {"": local_rank}
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config: LoraConfig | None
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=DEFAULT_LORA_TARGETS,
        )

    reward_base = code_execution_reward if reward_type == "dense" else code_binary_execution_reward
    reward_func = functools.partial(
        reward_base,
        execution_config={
            "timeout_seconds": timeout_seconds,
            "workers": execution_workers,
        },
    )
    args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=temperature,
        top_p=top_p,
        beta=beta,
        loss_type=loss_type,
        use_vllm=False,
        bf16=bf16,
        fp16=not bf16,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_first_step=True,
        log_completions=True,
        num_completions_to_print=2,
        save_strategy="steps",
        save_steps=max_steps,
        save_total_limit=1,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "final"))
    if int(os.environ.get("RANK", "0")) == 0:
        tokenizer.save_pretrained(output_dir / "final")

    payload = {
        "model_name_or_path": model_name_or_path,
        "adapter_path": adapter_path,
        "train_jsonl": str(train_jsonl),
        "train_jsonl_sha256": sha256_file(train_jsonl),
        "output_dir": str(output_dir),
        "records": len(rows),
        "max_steps": max_steps,
        "max_completion_length": max_completion_length,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_generations": num_generations,
        "temperature": temperature,
        "top_p": top_p,
        "beta": beta,
        "loss_type": loss_type,
        "reward_type": reward_type,
        "timeout_seconds": timeout_seconds,
        "execution_workers": execution_workers,
        "load_in_4bit": load_in_4bit,
        "bf16": bf16,
        "seed": seed,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(local_rank)
        if torch.cuda.is_available()
        else None,
        "metrics": train_result.metrics,
        "claim_boundary": "Training receipt only; model improvement requires held-out execution eval.",
    }
    if int(os.environ.get("RANK", "0")) == 0:
        (output_dir / "train_report.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return payload


def _load_grpo_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt = payload.get("messages") or payload.get("prompt")
            tests = payload.get("tests")
            if not prompt or not tests:
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "tests": tests,
                    "task_id": payload.get("task_id"),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no GRPO rows loaded from {path}")
    return rows
