from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def train_real_sft(
    model_name_or_path: str,
    train_jsonl: Path,
    output_dir: Path,
    max_steps: int = 5,
    max_length: int = 2048,
    limit: int | None = 64,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    completion_only_loss: bool = False,
    load_in_4bit: bool = False,
    bf16: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = _load_sft_rows(
        train_jsonl,
        tokenizer=tokenizer,
        limit=limit,
        completion_only_loss=completion_only_loss,
    )
    dataset = Dataset.from_list(rows)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_LORA_TARGETS,
    )
    args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        dataset_text_field="text",
        completion_only_loss=completion_only_loss,
        packing=False,
        bf16=bf16,
        fp16=not bf16,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=1,
        logging_first_step=True,
        save_steps=max_steps,
        save_total_limit=1,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(output_dir / "final")

    payload = {
        "model_name_or_path": model_name_or_path,
        "train_jsonl": str(train_jsonl),
        "train_jsonl_sha256": _sha256_file(train_jsonl),
        "output_dir": str(output_dir),
        "records": len(rows),
        "max_steps": max_steps,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "completion_only_loss": completion_only_loss,
        "load_in_4bit": load_in_4bit,
        "bf16": bf16,
        "seed": seed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "metrics": train_result.metrics,
    }
    (output_dir / "train_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sft_rows(
    path: Path,
    tokenizer: Any,
    limit: int | None,
    completion_only_loss: bool = False,
) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            payload = json.loads(line)
            messages = payload["messages"]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if completion_only_loss:
                if not messages or messages[-1].get("role") != "assistant":
                    raise ValueError("completion-only SFT rows must end with an assistant message")
                prompt = tokenizer.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                )
                if not full_text.startswith(prompt):
                    raise ValueError("chat template prompt is not a prefix of the full conversation")
                rows.append({"prompt": prompt, "completion": full_text[len(prompt) :]})
            else:
                rows.append({"text": full_text})
    if not rows:
        raise ValueError(f"no SFT rows loaded from {path}")
    return rows
