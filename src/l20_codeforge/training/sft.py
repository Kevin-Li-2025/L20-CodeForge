from __future__ import annotations

import json
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

    rows = _load_sft_rows(train_jsonl, tokenizer=tokenizer, limit=limit)
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
        model_kwargs["device_map"] = "auto"

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
        "output_dir": str(output_dir),
        "records": len(rows),
        "max_steps": max_steps,
        "max_length": max_length,
        "load_in_4bit": load_in_4bit,
        "metrics": train_result.metrics,
    }
    (output_dir / "train_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_sft_rows(path: Path, tokenizer: Any, limit: int | None) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            payload = json.loads(line)
            text = tokenizer.apply_chat_template(
                payload["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            rows.append({"text": text})
    if not rows:
        raise ValueError(f"no SFT rows loaded from {path}")
    return rows
