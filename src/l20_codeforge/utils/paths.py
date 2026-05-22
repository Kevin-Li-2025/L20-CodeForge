from __future__ import annotations

from pathlib import Path


PROJECT_DIRS = [
    "artifacts/checkpoints",
    "artifacts/eval_cards",
    "artifacts/reports",
    "artifacts/trajectories",
    "data/raw",
    "data/processed",
    "logs",
]


def ensure_project_dirs(root: Path) -> list[Path]:
    created = []
    for rel in PROJECT_DIRS:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created
