#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Solve the problem correctly, "
    "handle edge cases, and return a complete Python solution."
)


def normalize_for_overlap(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def load_lcb_exclusion_hashes(paths: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for key in ("question_content", "question_title"):
                    value = record.get(key)
                    if isinstance(value, str) and value.strip():
                        hashes.add(stable_hash(normalize_for_overlap(value)))
    return hashes


def row_to_sft_record(
    row: dict[str, Any],
    *,
    system_prompt: str,
    max_query_chars: int | None,
    max_response_chars: int | None,
    exclusion_hashes: set[str],
    keep_metadata: bool,
) -> dict[str, Any] | None:
    query = str(row.get("query") or row.get("prompt") or "").strip()
    response = str(row.get("response") or row.get("completion") or "").strip()
    if not query or not response:
        return None
    if stable_hash(normalize_for_overlap(query)) in exclusion_hashes:
        return None
    record: dict[str, Any] = {
        "dataset": "IIGroup/X-Coder-SFT-376k",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": truncate_text(query, max_query_chars)},
            {"role": "assistant", "content": truncate_text(response, max_response_chars) + "\n"},
        ],
    }
    if keep_metadata:
        record["metadata"] = {
            key: row[key]
            for key in sorted(row)
            if key not in {"query", "prompt", "response", "completion"}
        }
    return record


def iter_dataset_rows(
    dataset: str,
    split: str,
    streaming: bool,
) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset(dataset, split=split, streaming=streaming)
    for row in rows:
        yield dict(row)


def build_jsonl(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exclusion_paths = [Path(path) for path in args.exclude_lcb_jsonl]
    exclusion_hashes = load_lcb_exclusion_hashes(exclusion_paths)
    rng = random.Random(args.seed)

    rows_seen = 0
    rows_written = 0
    rows_skipped = 0
    reservoir: list[dict[str, Any]] = []

    for row in iter_dataset_rows(args.dataset, args.split, args.streaming):
        rows_seen += 1
        record = row_to_sft_record(
            row,
            system_prompt=args.system_prompt,
            max_query_chars=args.max_query_chars,
            max_response_chars=args.max_response_chars,
            exclusion_hashes=exclusion_hashes,
            keep_metadata=args.keep_metadata,
        )
        if record is None:
            rows_skipped += 1
            continue
        if args.shuffle_buffer <= 0:
            reservoir.append(record)
        else:
            if len(reservoir) < args.shuffle_buffer:
                reservoir.append(record)
            else:
                index = rng.randrange(rows_seen)
                if index < args.shuffle_buffer:
                    reservoir[index] = record
        if args.limit is not None and len(reservoir) >= args.limit and args.shuffle_buffer <= 0:
            break
        if args.max_source_rows is not None and rows_seen >= args.max_source_rows:
            break

    if args.shuffle_buffer > 0:
        rng.shuffle(reservoir)
    if args.limit is not None:
        reservoir = reservoir[: args.limit]

    with output.open("w", encoding="utf-8") as handle:
        for record in reservoir:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            rows_written += 1

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "output": str(output),
        "records": rows_written,
        "rows_seen": rows_seen,
        "rows_skipped": rows_skipped,
        "limit": args.limit,
        "max_source_rows": args.max_source_rows,
        "streaming": args.streaming,
        "shuffle_buffer": args.shuffle_buffer,
        "seed": args.seed,
        "exclude_lcb_jsonl": [str(path) for path in exclusion_paths],
        "exclusion_hash_count": len(exclusion_hashes),
        "license_note": (
            "X-Coder-SFT-376k dataset card lists MIT license; verify before "
            "redistribution."
        ),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build chat SFT JSONL from the public X-Coder competitive-programming dataset."
    )
    parser.add_argument("--dataset", default="IIGroup/X-Coder-SFT-376k")
    parser.add_argument("--split", default="unique_prompt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--max-source-rows", type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-query-chars", type=int, default=24000)
    parser.add_argument("--max-response-chars", type=int, default=48000)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--exclude-lcb-jsonl", action="append", default=[])
    parser.add_argument("--keep-metadata", action="store_true")
    return parser


def main() -> None:
    build_jsonl(build_parser().parse_args())


if __name__ == "__main__":
    main()
