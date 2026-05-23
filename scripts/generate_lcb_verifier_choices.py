#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lcb_expected_output_verifier_prompts import parse_choice_payload  # noqa: E402
from build_lcb_behavior_test_prompts import sha256_file  # noqa: E402
from generate_lcb_behavior_tests import display_model_name, load_model  # noqa: E402


SYSTEM_MESSAGE = (
    "You are an expected-output verifier for programming contest problems. "
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
            record_id = record.get("record_id")
            if isinstance(record_id, str):
                records[record_id] = record
    return records


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


def generate_verifier_one(
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


def parse_output_records(
    output_records: list[dict[str, Any]],
    prompt_records_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    parsed = []
    for output in output_records:
        record_id = output.get("record_id")
        prompt_record = prompt_records_by_id.get(record_id)
        if prompt_record is None:
            continue
        try:
            choice = parse_choice_payload(str(output.get("raw_output") or ""))
        except ValueError:
            choice = {"choice": "NONE", "confidence": 0.0, "reason": "parse_error"}
        parsed.append(
            {
                "record_id": record_id,
                "question_id": prompt_record["question_id"],
                "input_index": prompt_record["input_index"],
                **choice,
            }
        )
    return parsed


def write_choices(
    path: Path,
    parsed_records: list[dict[str, Any]],
    model: str,
    prompts_path: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "records": parsed_records,
                "metadata": {
                    "source": "local_model_lcb_expected_output_choice_verifier_v1",
                    "model": display_model_name(model),
                    "prompts": str(prompts_path),
                    "prompts_sha256": sha256_file(prompts_path),
                    "records_total": len(parsed_records),
                },
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate expected-output verifier choices from LCB prompt records."
    )
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prompts_path = Path(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = output_dir / "llm_outputs.jsonl"
    choices_path = output_dir / "verifier_choices.json"

    prompt_records = load_prompt_records(prompts_path, args.limit)
    existing = load_existing_outputs(outputs_path) if args.resume else {}
    manifest = {
        "prompts": str(prompts_path),
        "prompts_sha256": sha256_file(prompts_path),
        "prompt_count": len(prompt_records),
        "already_generated": len(existing),
        "model": display_model_name(args.model),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
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
            record_id = record["record_id"]
            if record_id in existing:
                print(f"[{index}/{len(prompt_records)}] resume {record_id}")
                continue
            print(f"[{index}/{len(prompt_records)}] generate {record_id}")
            raw_output = generate_verifier_one(
                model=model,
                tokenizer=tokenizer,
                prompt=record["prompt"],
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            output_record = {
                "record_id": record_id,
                "question_id": record["question_id"],
                "input_index": record["input_index"],
                "raw_output": raw_output,
            }
            output_records.append(output_record)
            handle.write(json.dumps(output_record, ensure_ascii=True) + "\n")
            handle.flush()

    parsed_records = parse_output_records(
        output_records,
        {record["record_id"]: record for record in prompt_records},
    )
    write_choices(
        choices_path,
        parsed_records=parsed_records,
        model=args.model,
        prompts_path=prompts_path,
    )
    manifest.update(
        {
            "outputs": str(outputs_path),
            "verifier_choices": str(choices_path),
            "generated_records": len(output_records),
            "parsed_records": len(parsed_records),
            "seconds": round(time.monotonic() - started, 3),
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
