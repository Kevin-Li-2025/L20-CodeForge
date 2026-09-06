from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def load_interpolation_module():
    path = Path(__file__).parents[1] / "scripts" / "interpolate_lora_adapters.py"
    spec = importlib.util.spec_from_file_location("interpolate_lora_adapters", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_adapter(path: Path, value: float) -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "model",
                "peft_type": "LORA",
                "r": 2,
                "lora_alpha": 4,
                "target_modules": ["q_proj"],
                "rank_pattern": {},
                "alpha_pattern": {},
                "use_rslora": False,
                "use_dora": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    safetensors_torch.save_file(
        {
            "layer.lora_A.weight": torch.full((2, 2), value),
            "layer.lora_B.weight": torch.full((2, 2), value + 1),
        },
        path / "adapter_model.safetensors",
    )


def test_interpolates_compatible_lora_tensors(tmp_path: Path) -> None:
    module = load_interpolation_module()
    start = tmp_path / "start"
    end = tmp_path / "end"
    output = tmp_path / "output"
    write_adapter(start, 0.0)
    write_adapter(end, 4.0)

    report = module.interpolate_lora_adapters(start, end, output, 0.25)
    state = safetensors_torch.load_file(output / "adapter_model.safetensors")

    assert torch.equal(state["layer.lora_A.weight"], torch.full((2, 2), 1.0))
    assert torch.equal(state["layer.lora_B.weight"], torch.full((2, 2), 2.0))
    assert report["interpolation_fraction"] == 0.25
    assert report["changed_tensors_vs_start"] == 2


def test_rejects_structurally_different_adapters(tmp_path: Path) -> None:
    module = load_interpolation_module()
    start = tmp_path / "start"
    end = tmp_path / "end"
    write_adapter(start, 0.0)
    write_adapter(end, 4.0)
    config_path = end / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["r"] = 8
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="adapter structures differ"):
        module.interpolate_lora_adapters(start, end, tmp_path / "output", 0.5)


def test_accepts_target_modules_in_different_order(tmp_path: Path) -> None:
    module = load_interpolation_module()
    start = tmp_path / "start"
    end = tmp_path / "end"
    write_adapter(start, 0.0)
    write_adapter(end, 4.0)
    config_path = end / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["target_modules"] = ["v_proj", "q_proj"]
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    start_config_path = start / "adapter_config.json"
    start_config = json.loads(start_config_path.read_text())
    start_config["target_modules"] = ["q_proj", "v_proj"]
    start_config_path.write_text(json.dumps(start_config) + "\n", encoding="utf-8")

    report = module.interpolate_lora_adapters(start, end, tmp_path / "output", 0.5)

    assert report["tensor_count"] == 2
