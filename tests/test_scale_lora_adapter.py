from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_scaling_module():
    path = Path(__file__).parents[1] / "scripts" / "scale_lora_adapter.py"
    spec = importlib.util.spec_from_file_location("scale_lora_adapter", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scale_lora_adapter_changes_only_global_alpha(tmp_path: Path) -> None:
    module = load_scaling_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "alpha_pattern": {},
                "rank_pattern": {},
                "use_rslora": False,
                "use_dora": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    weights = b"synthetic BF16 safetensors placeholder"
    (source / "adapter_model.safetensors").write_bytes(weights)
    output = tmp_path / "scaled"

    report = module.scale_lora_adapter(source, output, 0.5)
    scaled_config = json.loads((output / "adapter_config.json").read_text())

    assert (output / "adapter_model.safetensors").read_bytes() == weights
    assert scaled_config["lora_alpha"] == 16
    assert scaled_config["r"] == 16
    assert report["scale"] == 0.5
    assert report["source_lora_alpha"] == 32
    assert report["output_lora_alpha"] == 16
