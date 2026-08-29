#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

STRUCTURAL_CONFIG_KEYS = (
    "base_model_name_or_path",
    "peft_type",
    "r",
    "lora_alpha",
    "target_modules",
    "rank_pattern",
    "alpha_pattern",
    "use_rslora",
    "use_dora",
)


def normalized_config_value(key: str, value: Any) -> Any:
    if key == "target_modules" and isinstance(value, list):
        return sorted(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interpolate_lora_adapters(
    start: Path,
    end: Path,
    output: Path,
    interpolation_fraction: float,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    if not 0.0 < interpolation_fraction < 1.0:
        raise ValueError("interpolation_fraction must satisfy 0 < fraction < 1")
    if output.exists():
        raise FileExistsError(output)

    start_config_path = start / "adapter_config.json"
    end_config_path = end / "adapter_config.json"
    start_weights_path = start / "adapter_model.safetensors"
    end_weights_path = end / "adapter_model.safetensors"
    for path in (
        start_config_path,
        end_config_path,
        start_weights_path,
        end_weights_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    start_config = json.loads(start_config_path.read_text(encoding="utf-8"))
    end_config = json.loads(end_config_path.read_text(encoding="utf-8"))
    if start_config.get("peft_type") != "LORA" or end_config.get("peft_type") != "LORA":
        raise ValueError("both adapters must have peft_type=LORA")
    mismatched_config = {
        key: {"start": start_config.get(key), "end": end_config.get(key)}
        for key in STRUCTURAL_CONFIG_KEYS
        if normalized_config_value(key, start_config.get(key))
        != normalized_config_value(key, end_config.get(key))
    }
    if mismatched_config:
        raise ValueError(f"adapter structures differ: {mismatched_config}")

    start_state = load_file(start_weights_path, device="cpu")
    end_state = load_file(end_weights_path, device="cpu")
    if start_state.keys() != end_state.keys():
        missing = sorted(start_state.keys() - end_state.keys())
        extra = sorted(end_state.keys() - start_state.keys())
        raise ValueError(f"adapter tensor keys differ: missing={missing}, extra={extra}")

    output_state: dict[str, torch.Tensor] = {}
    changed_tensors = 0
    for key in sorted(start_state):
        start_tensor = start_state[key]
        end_tensor = end_state[key]
        if start_tensor.shape != end_tensor.shape or start_tensor.dtype != end_tensor.dtype:
            raise ValueError(
                f"tensor structure differs for {key}: "
                f"{start_tensor.shape}/{start_tensor.dtype} vs "
                f"{end_tensor.shape}/{end_tensor.dtype}"
            )
        if start_tensor.is_floating_point():
            interpolated = torch.lerp(
                start_tensor.float(),
                end_tensor.float(),
                interpolation_fraction,
            ).to(start_tensor.dtype)
            output_state[key] = interpolated.contiguous()
            changed_tensors += int(not torch.equal(start_tensor, interpolated))
        else:
            if not torch.equal(start_tensor, end_tensor):
                raise ValueError(f"non-floating tensor differs for {key}")
            output_state[key] = start_tensor.contiguous()

    with safe_open(end_weights_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    shutil.copytree(start, output)
    output_weights_path = output / "adapter_model.safetensors"
    save_file(output_state, output_weights_path, metadata=metadata)
    report = {
        "start": str(start),
        "end": str(end),
        "output": str(output),
        "interpolation_fraction": interpolation_fraction,
        "method": "parameter-space linear interpolation from verified SFT to final RLVR",
        "tensor_count": len(output_state),
        "changed_tensors_vs_start": changed_tensors,
        "start_weights_sha256": sha256_file(start_weights_path),
        "end_weights_sha256": sha256_file(end_weights_path),
        "output_weights_sha256": sha256_file(output_weights_path),
        "adapter_config_sha256": sha256_file(output / "adapter_config.json"),
        "claim_boundary": (
            "Deterministic adapter construction only; improvement requires frozen-dev evaluation."
        ),
    }
    (output / "interpolation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=Path)
    parser.add_argument("end", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fraction", type=float, required=True)
    args = parser.parse_args()
    report = interpolate_lora_adapters(args.start, args.end, args.output, args.fraction)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
