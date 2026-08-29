#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scale_lora_adapter(source: Path, output: Path, scale: float) -> dict[str, object]:
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must satisfy 0 < scale <= 1")
    source_weights = source / "adapter_model.safetensors"
    source_config = source / "adapter_config.json"
    if not source_weights.exists():
        raise FileNotFoundError(source_weights)
    if not source_config.exists():
        raise FileNotFoundError(source_config)
    if output.exists():
        raise FileExistsError(output)

    config = json.loads(source_config.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise ValueError("adapter must have peft_type=LORA")
    if config.get("alpha_pattern") or config.get("rank_pattern"):
        raise ValueError("per-module alpha/rank patterns are not supported")
    if config.get("use_rslora") or config.get("use_dora"):
        raise ValueError("RSLoRA and DoRA adapters are not supported")
    source_alpha = int(config["lora_alpha"])
    target_alpha_float = source_alpha * scale
    target_alpha = round(target_alpha_float)
    if abs(target_alpha - target_alpha_float) > 1e-12 or target_alpha <= 0:
        raise ValueError("scaled lora_alpha must be a positive integer")

    shutil.copytree(source, output)
    config["lora_alpha"] = target_alpha
    output_config = output / "adapter_config.json"
    output_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    report: dict[str, object] = {
        "source": str(source),
        "output": str(output),
        "scale": scale,
        "method": "scale global lora_alpha; LoRA delta is proportional to alpha/r",
        "source_lora_alpha": source_alpha,
        "output_lora_alpha": target_alpha,
        "source_weights_sha256": sha256_file(source_weights),
        "output_weights_sha256": sha256_file(output / "adapter_model.safetensors"),
        "source_config_sha256": sha256_file(source_config),
        "output_config_sha256": sha256_file(output_config),
    }
    (output / "scaling_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    report = scale_lora_adapter(args.source, args.output, args.scale)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
