from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def read_jsonl(path: Path, model: type[ModelT]) -> Iterator[ModelT]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield model.model_validate_json(stripped)
            except ValueError as exc:
                raise ValueError(f"invalid JSONL record at {path}:{line_number}") from exc


def write_json(path: Path, payload: BaseModel | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        text = payload.model_dump_json(indent=2)
    else:
        text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[BaseModel | dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if isinstance(record, BaseModel):
                handle.write(record.model_dump_json() + "\n")
            else:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count

