#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import guard for setup diagnostics
        print(json.dumps({"ok": False, "error": f"torch import failed: {exc}"}, indent=2))
        return 1

    payload: dict[str, object] = {
        "ok": True,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        torch.set_float32_matmul_precision("high")
        x = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
        y = x @ x.T
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info(device)
        payload.update(
            {
                "device_name": props.name,
                "total_vram_gib": round(total / 1024**3, 2),
                "free_vram_gib": round(free / 1024**3, 2),
                "bf16_matmul_mean": float(y.float().mean().detach().cpu()),
            }
        )

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["cuda_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

