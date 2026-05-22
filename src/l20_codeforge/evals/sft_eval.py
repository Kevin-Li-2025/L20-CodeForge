from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_sft_eval_rows(
    path: Path,
    limit: int | None = None,
    exclude_instance_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exclude_instance_ids = exclude_instance_ids or set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("instance_id") in exclude_instance_ids:
                continue
            rows.append(payload)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no eval rows loaded from {path}")
    return rows


def load_instance_id_set(path: Path) -> set[str]:
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            instance_id = payload.get("instance_id")
            if isinstance(instance_id, str):
                ids.add(instance_id)
    return ids


def is_unified_diff_like(text: str) -> bool:
    return "diff --git " in text or ("--- " in text and "+++ " in text and "@@" in text)


def evaluate_real_sft_model(
    model_name_or_path: str,
    eval_jsonl: Path,
    output: Path,
    adapter_path: str | None = None,
    exclude_jsonl: Path | None = None,
    limit: int | None = 50,
    max_length: int = 4096,
    max_new_tokens: int = 384,
    generate_samples: int = 3,
    load_in_4bit: bool = True,
    bf16: bool = True,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    exclude_ids = load_instance_id_set(exclude_jsonl) if exclude_jsonl else set()
    rows = load_sft_eval_rows(eval_jsonl, limit=limit, exclude_instance_ids=exclude_ids)

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name_or_path,
        trust_remote_code=True,
        local_files_only=True,
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

    total_loss = 0.0
    total_tokens = 0
    per_record = []
    with torch.inference_mode():
        for row in rows:
            inputs, labels, assistant_tokens = _encode_assistant_loss_example(
                tokenizer=tokenizer,
                messages=row["messages"],
                max_length=max_length,
                device=model.device,
            )
            if assistant_tokens == 0:
                continue
            result = model(**inputs, labels=labels)
            loss = float(result.loss.detach().cpu())
            total_loss += loss * assistant_tokens
            total_tokens += assistant_tokens
            per_record.append(
                {
                    "dataset": row.get("dataset"),
                    "instance_id": row.get("instance_id"),
                    "repo": row.get("repo"),
                    "assistant_tokens": assistant_tokens,
                    "loss": loss,
                }
            )

    if total_tokens == 0:
        raise ValueError("no assistant tokens available for evaluation")

    generations = []
    for row in rows[:generate_samples]:
        prompt_messages = row["messages"][:-1]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        generations.append(
            {
                "dataset": row.get("dataset"),
                "instance_id": row.get("instance_id"),
                "repo": row.get("repo"),
                "unified_diff_like": is_unified_diff_like(generated),
                "generated": generated,
                "gold_prefix": row["messages"][-1]["content"][:1000],
            }
        )

    mean_loss = total_loss / total_tokens
    payload = {
        "model_name_or_path": model_name_or_path,
        "adapter_path": adapter_path,
        "eval_jsonl": str(eval_jsonl),
        "exclude_jsonl": str(exclude_jsonl) if exclude_jsonl else None,
        "records": len(per_record),
        "assistant_tokens": total_tokens,
        "max_length": max_length,
        "mean_assistant_nll": mean_loss,
        "perplexity": float(torch.exp(torch.tensor(mean_loss)).item()),
        "unified_diff_like_generations": sum(
            1 for sample in generations if sample["unified_diff_like"]
        ),
        "generations": generations,
        "per_record": per_record,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _encode_assistant_loss_example(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int,
    device: Any,
) -> tuple[dict[str, Any], Any, int]:
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_encoded = tokenizer(
        full,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    prompt_encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {key: value.to(device) for key, value in full_encoded.items()}
    labels = inputs["input_ids"].clone()
    prompt_len = min(prompt_encoded["input_ids"].shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100
    assistant_tokens = int((labels != -100).sum().item())
    return inputs, labels, assistant_tokens
