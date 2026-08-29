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
    replay_jsonl: Path | None = None,
    replay_loss_weight: float = 0.0,
    replay_max_length: int = 3072,
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
    """Run executable-reward GRPO with optional retention replay cross-entropy."""

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    if reward_type not in {"dense", "binary"}:
        raise ValueError("reward_type must be 'dense' or 'binary'")
    if replay_loss_weight < 0:
        raise ValueError("replay_loss_weight must be non-negative")
    if replay_loss_weight > 0 and replay_jsonl is None:
        raise ValueError("replay_jsonl is required when replay_loss_weight is positive")
    rows = _load_grpo_rows(train_jsonl, limit=limit)
    replay_rows: list[dict[str, Any]] = []
    if replay_jsonl is not None:
        replay_rows = _load_replay_rows(replay_jsonl)
        rows = _attach_replay_rows(rows, replay_rows, seed=seed)
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
    trainer_class = _replay_trainer_class(GRPOTrainer)
    trainer = trainer_class(
        model=model,
        reward_funcs=reward_func,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        replay_loss_weight=replay_loss_weight,
        replay_max_length=replay_max_length,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "final"))
    if int(os.environ.get("RANK", "0")) == 0:
        tokenizer.save_pretrained(output_dir / "final")

    payload = {
        "model_name_or_path": model_name_or_path,
        "adapter_path": adapter_path,
        "replay_jsonl": str(replay_jsonl) if replay_jsonl else None,
        "replay_jsonl_sha256": sha256_file(replay_jsonl) if replay_jsonl else None,
        "replay_records": len(replay_rows),
        "replay_loss_weight": replay_loss_weight,
        "replay_max_length": replay_max_length,
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


def _replay_trainer_class(base_class: type[Any]) -> type[Any]:
    """Create a GRPOTrainer subclass without importing optional train deps at module load."""

    class ReplayGRPOTrainer(base_class):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *args: Any,
            replay_loss_weight: float = 0.0,
            replay_max_length: int = 3072,
            **kwargs: Any,
        ) -> None:
            self.replay_loss_weight = replay_loss_weight
            self.replay_max_length = replay_max_length
            super().__init__(*args, **kwargs)

        def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
            output = super()._generate_and_score_completions(inputs)
            if self.replay_loss_weight == 0.0:
                return output
            messages = [row.get("replay_messages") for row in inputs]
            if not all(messages):
                raise ValueError("every GRPO row must carry replay_messages")
            replay_batch = _tokenize_replay_messages(
                self.processing_class,
                messages,
                max_length=self.replay_max_length,
            )
            device = self.accelerator.device
            output.update({key: value.to(device) for key, value in replay_batch.items()})
            return output

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            import torch

            loss = super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            if self.replay_loss_weight == 0.0:
                return loss
            replay_outputs = model(
                input_ids=inputs["replay_input_ids"],
                attention_mask=inputs["replay_attention_mask"],
                labels=inputs["replay_labels"],
                use_cache=False,
            )
            replay_loss = replay_outputs.loss
            mode = "train" if model.training else "eval"
            gathered = self.accelerator.gather(replay_loss.detach()).mean().item()
            self._metrics[mode]["replay_loss"].append(gathered)
            weighted = (
                self.replay_loss_weight
                * replay_loss
                / max(1, self.current_gradient_accumulation_steps)
            )
            if not torch.isfinite(weighted):
                raise FloatingPointError("non-finite replay loss")
            return loss + weighted

    return ReplayGRPOTrainer


def _tokenize_replay_messages(
    tokenizer: Any,
    rows: list[list[dict[str, str]]],
    *,
    max_length: int,
) -> dict[str, Any]:
    """Tokenize replay chats with prompt labels masked and deterministic left padding."""

    import torch

    if max_length <= 0:
        raise ValueError("replay_max_length must be positive")
    encoded: list[tuple[list[int], int]] = []
    for messages in rows:
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("replay rows must end with an assistant message")
        full_ids = list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        prompt_ids = list(
            tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("replay prompt tokens are not a prefix of full chat tokens")
        overflow = max(0, len(full_ids) - max_length)
        full_ids = full_ids[overflow:]
        prompt_length = max(0, len(prompt_ids) - overflow)
        if prompt_length >= len(full_ids):
            raise ValueError("replay truncation removed the entire assistant completion")
        encoded.append((full_ids, prompt_length))

    width = max(len(ids) for ids, _ in encoded)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("replay tokenizer requires a pad token")
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for ids, prompt_length in encoded:
        padding = width - len(ids)
        input_ids.append([pad_token_id] * padding + ids)
        attention_mask.append([0] * padding + [1] * len(ids))
        labels.append([-100] * (padding + prompt_length) + ids[prompt_length:])
    return {
        "replay_input_ids": torch.tensor(input_ids, dtype=torch.long),
        "replay_attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "replay_labels": torch.tensor(labels, dtype=torch.long),
    }


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


def _load_replay_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            messages = payload.get("messages")
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError("replay JSONL rows must end with an assistant message")
            rows.append(payload)
    if not rows:
        raise ValueError(f"no replay rows loaded from {path}")
    return rows


def _attach_replay_rows(
    grpo_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        replay_rows,
        key=lambda row: _stable_json_hash({"seed": seed, "messages": row["messages"]}),
    )
    return [
        {**row, "replay_messages": ordered[index % len(ordered)]["messages"]}
        for index, row in enumerate(grpo_rows)
    ]


def _stable_json_hash(payload: Any) -> str:
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
