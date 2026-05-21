#!/usr/bin/env python3
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a model snapshot into HF cache.")
    parser.add_argument("model_id")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--allow",
        nargs="*",
        default=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.tiktoken"],
        help="Allow patterns passed to snapshot_download.",
    )
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        allow_patterns=args.allow,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

