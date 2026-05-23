#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lcb_behavior_test_prompts import coerce_inputs, extract_json_payload, sha256_file


SYSTEM_MESSAGE = (
    "You generate adversarial input-only tests for programming contest problems. "
    "Return valid JSON only."
)


def load_prompt_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def load_existing_outputs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            question_id = record.get("question_id")
            if isinstance(question_id, str):
                records[question_id] = record
    return records


def parse_behavior_inputs_from_outputs(
    output_records: list[dict[str, Any]],
    max_inputs: int,
    max_input_chars: int,
) -> list[dict[str, Any]]:
    parsed = []
    for record in output_records:
        question_id = record.get("question_id")
        raw_output = record.get("raw_output")
        if not isinstance(question_id, str) or not isinstance(raw_output, str):
            continue
        try:
            payload = extract_json_payload(raw_output)
            inputs = coerce_inputs(payload, max_inputs=max_inputs, max_input_chars=max_input_chars)
        except ValueError:
            inputs = []
        parsed.append(
            {
                "question_id": question_id,
                "inputs": inputs,
                "n_inputs": len(inputs),
                "source": "local_model_candidate_aware_differential_v1",
            }
        )
    return parsed


def write_behavior_inputs(
    path: Path,
    parsed_records: list[dict[str, Any]],
    model: str,
    prompts_path: Path,
) -> None:
    nonempty = [record for record in parsed_records if record["inputs"]]
    path.write_text(
        json.dumps(
            {
                "records": nonempty,
                "metadata": {
                    "source": "local_model_candidate_aware_differential_v1",
                    "model": model,
                    "prompts": str(prompts_path),
                    "prompts_sha256": sha256_file(prompts_path),
                    "records_total": len(parsed_records),
                    "records_nonempty": len(nonempty),
                    "inputs_total": sum(record["n_inputs"] for record in parsed_records),
                },
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_model(model_name_or_path: str, load_in_4bit: bool, bf16: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    local_files_only = Path(model_name_or_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
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
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
        **model_kwargs,
    )
    model.eval()
    return model, tokenizer


def render_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{SYSTEM_MESSAGE}\n\n{prompt}\n\nJSON:"


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    max_input_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    rendered = render_prompt(tokenizer, prompt)
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
        kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        kwargs["do_sample"] = False
    with torch.inference_mode():
        output_ids = model.generate(**encoded, **kwargs)
    prompt_tokens = encoded["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate candidate-aware LCB behavior tests from prompt records."
    )
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-inputs", type=int, default=12)
    parser.add_argument("--max-input-chars", type=int, default=20000)
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = output_dir / "llm_outputs.jsonl"
    behavior_inputs_path = output_dir / "behavior_inputs.json"

    prompt_records = load_prompt_records(prompts_path, args.limit)
    existing = load_existing_outputs(outputs_path) if args.resume else {}
    if args.dry_run:
        manifest = {
            "prompts": str(prompts_path),
            "prompts_sha256": sha256_file(prompts_path),
            "prompt_count": len(prompt_records),
            "already_generated": len(existing),
            "model": args.model,
            "dry_run": True,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=True))
        return

    model, tokenizer = load_model(
        model_name_or_path=args.model,
        load_in_4bit=not args.no_4bit,
        bf16=not args.no_bf16,
    )

    started = time.monotonic()
    output_records: list[dict[str, Any]] = []
    if args.resume:
        output_records.extend(existing.values())
    with outputs_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for index, record in enumerate(prompt_records, start=1):
            question_id = record["question_id"]
            if question_id in existing:
                print(f"[{index}/{len(prompt_records)}] resume {question_id}")
                continue
            print(f"[{index}/{len(prompt_records)}] generate {question_id}")
            raw_output = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=record["prompt"],
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            output_record = {
                "question_id": question_id,
                "question_title": record.get("question_title"),
                "raw_output": raw_output,
            }
            output_records.append(output_record)
            handle.write(json.dumps(output_record, ensure_ascii=True) + "\n")
            handle.flush()

    parsed_records = parse_behavior_inputs_from_outputs(
        output_records,
        max_inputs=args.max_inputs,
        max_input_chars=args.max_input_chars,
    )
    write_behavior_inputs(
        behavior_inputs_path,
        parsed_records=parsed_records,
        model=args.model,
        prompts_path=prompts_path,
    )
    manifest = {
        "prompts": str(prompts_path),
        "prompts_sha256": sha256_file(prompts_path),
        "outputs": str(outputs_path),
        "behavior_inputs": str(behavior_inputs_path),
        "model": args.model,
        "prompt_count": len(prompt_records),
        "generated_records": len(output_records),
        "nonempty_behavior_records": sum(bool(record["inputs"]) for record in parsed_records),
        "behavior_inputs_total": sum(record["n_inputs"] for record in parsed_records),
        "seconds": round(time.monotonic() - started, 3),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
