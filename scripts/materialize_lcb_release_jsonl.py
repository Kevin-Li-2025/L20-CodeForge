#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


RELEASE_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
    "release_latest": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else output_jsonl.with_suffix(".manifest.json")
    release_files = RELEASE_FILES.get(args.release_version)
    if release_files is None:
        raise ValueError(
            f"unsupported release_version={args.release_version}; "
            f"expected one of {sorted(RELEASE_FILES)}"
        )

    started = time.monotonic()
    rows_written = 0
    shards: list[dict[str, Any]] = []
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for filename in release_files:
            shard_path = Path(
                hf_hub_download(
                    repo_id=args.dataset,
                    filename=filename,
                    repo_type="dataset",
                )
            )
            shard_rows = 0
            with shard_path.open(encoding="utf-8") as shard_handle:
                for line in shard_handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    handle.write(json.dumps(make_json_safe(row), ensure_ascii=True) + "\n")
                    rows_written += 1
                    shard_rows += 1
            shards.append(
                {
                    "filename": filename,
                    "rows": shard_rows,
                    "sha256": sha256_file(shard_path),
                }
            )

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "release_version": args.release_version,
        "output_jsonl": str(output_jsonl),
        "rows": rows_written,
        "sha256": sha256_file(output_jsonl),
        "shards": shards,
        "seconds": round(time.monotonic() - started, 3),
        "contains_private_tests": True,
        "storage_policy": "Do not commit this full hidden-test JSONL; commit only hashes and eval summaries.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a LiveCodeBench code_generation_lite release to JSONL."
    )
    parser.add_argument("--dataset", default="livecodebench/code_generation_lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--manifest")
    return parser


def main() -> None:
    materialize(build_parser().parse_args())


if __name__ == "__main__":
    main()
