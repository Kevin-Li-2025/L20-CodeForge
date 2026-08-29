from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from l20_codeforge.data.code_rlvr import (
    DEFAULT_SYSTEM_PROMPT,
    sha256_file,
    stable_hash,
    write_jsonl,
)
from l20_codeforge.rewards.code_execution import extract_python_code

MBPP_REPOSITORY = "google-research/google-research"
MBPP_REVISION = "041338718b4e8151372fd63677104c65b73a0a4e"
MBPP_RAW_SHA256 = "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"
MBPP_LICENSE = "CC-BY-4.0"
MBPP_RAW_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    f"{MBPP_REVISION}/mbpp/mbpp.jsonl"
)
FUNCTION_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Return only valid Python code that defines the "
    "requested function. Do not include Markdown fences or explanatory prose."
)


def materialize_mbpp_replay(
    output_dir: Path,
    *,
    rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Freeze disjoint official MBPP train replay and validation retention tasks.

    The original MBPP split contract reserves task ids 11--510 for test, 511--600
    for validation, and 601--974 for training. This builder never admits test ids.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "mbpp-source.jsonl"
    if rows is None:
        with urllib.request.urlopen(MBPP_RAW_URL, timeout=90) as response:
            source_bytes = response.read()
        source_path.write_bytes(source_bytes)
        source_sha256 = sha256_file(source_path)
        if source_sha256 != MBPP_RAW_SHA256:
            raise ValueError(
                "MBPP source hash mismatch: "
                f"expected {MBPP_RAW_SHA256}, got {source_sha256}"
            )
        rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines()]
    else:
        frozen_rows = [dict(row) for row in rows]
        write_jsonl(source_path, frozen_rows)
        source_sha256 = sha256_file(source_path)
        rows = frozen_rows

    train_records: list[dict[str, Any]] = []
    validation_tasks: list[dict[str, Any]] = []
    rejected_test_ids: list[int] = []
    for raw in rows:
        task_id = int(raw["task_id"])
        if 11 <= task_id <= 510:
            rejected_test_ids.append(task_id)
            continue
        prompt = str(raw["text"]).strip()
        code = str(raw["code"]).strip()
        tests = [str(test) for test in raw.get("test_list", [])]
        setup = str(raw.get("test_setup_code") or "")
        user_prompt = _mbpp_user_prompt(prompt, tests)
        if 601 <= task_id <= 974:
            train_records.append(
                {
                    "dataset": "mbpp-official-train-replay",
                    "messages": [
                        {"role": "system", "content": FUNCTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": code.rstrip() + "\n"},
                    ],
                    "metadata": {
                        "task_id": f"mbpp-train/{task_id}",
                        "source_task_id": task_id,
                        "split": "train",
                        "test_count": len(tests),
                        "source_revision": MBPP_REVISION,
                        "license": MBPP_LICENSE,
                    },
                }
            )
        elif 511 <= task_id <= 600:
            validation_tasks.append(
                {
                    "task_id": f"mbpp-validation/{task_id}",
                    "prompt": user_prompt,
                    "messages": [
                        {"role": "system", "content": FUNCTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "test_setup_code": setup,
                    "tests": tests,
                    "prompt_sha256": stable_hash(user_prompt),
                    "tests_sha256": stable_hash(json.dumps(tests, sort_keys=True)),
                    "source": {
                        "dataset": MBPP_REPOSITORY,
                        "revision": MBPP_REVISION,
                        "split": "validation",
                        "license": MBPP_LICENSE,
                    },
                }
            )

    train_records.sort(key=lambda row: int(row["metadata"]["source_task_id"]))
    validation_tasks.sort(key=lambda row: int(str(row["task_id"]).split("/")[-1]))
    if len(train_records) != 374 or len(validation_tasks) != 90:
        raise ValueError(
            "unexpected MBPP split sizes: "
            f"train={len(train_records)}/374 validation={len(validation_tasks)}/90"
        )

    train_path = output_dir / "train-replay.jsonl"
    validation_path = output_dir / "validation-retention.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(validation_path, validation_tasks)
    manifest = {
        "dataset": MBPP_REPOSITORY,
        "revision": MBPP_REVISION,
        "license": MBPP_LICENSE,
        "source_url": MBPP_RAW_URL,
        "source": str(source_path),
        "source_sha256": source_sha256,
        "train_id_range": [601, 974],
        "validation_id_range": [511, 600],
        "excluded_test_id_range": [11, 510],
        "excluded_test_ids_seen": len(rejected_test_ids),
        "train_records": len(train_records),
        "validation_tasks": len(validation_tasks),
        "train_replay": str(train_path),
        "train_replay_sha256": sha256_file(train_path),
        "validation_retention": str(validation_path),
        "validation_retention_sha256": sha256_file(validation_path),
        "claim_boundary": (
            "Only official MBPP train references enter optimization. Validation tasks are "
            "retention development data. MBPP test and MBPP+ are excluded."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_lcb_verified_trajectory_sft(
    eval_all_json: Path,
    output: Path,
    *,
    max_contest_date: str,
    min_contest_date: str | None = None,
    max_records: int = 138,
    seed: int = 20260830,
) -> dict[str, Any]:
    """Distill full-harness-passing historical L20 system outputs.

    ``eval_all_json`` must be a system-selected n=1 replay such as the public-test
    selected L20 artifact. Only dates inside the explicit historical training
    window are admitted, so later LiveCodeBench tasks remain evaluation-only.
    """

    rows = json.loads(eval_all_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("LiveCodeBench eval_all must contain a JSON list")
    max_date = datetime.fromisoformat(max_contest_date)
    min_date = datetime.fromisoformat(min_contest_date) if min_contest_date else None
    admitted: list[dict[str, Any]] = []
    for row in rows:
        contest_date = datetime.fromisoformat(str(row["contest_date"]))
        if contest_date > max_date or (min_date is not None and contest_date < min_date):
            continue
        codes = [str(code) for code in row.get("code_list", [])]
        grades = [bool(value) for value in row.get("graded_list", [])]
        if len(codes) != len(grades):
            raise ValueError(f"misaligned LCB code/grade lists for {row.get('question_id')}")
        passing = [extract_python_code(code) for code, passed in zip(codes, grades) if passed]
        if not passing:
            continue
        code = min(passing, key=lambda value: (len(value), value))
        question_id = str(row["question_id"])
        admitted.append(
            {
                "dataset": "l20-livecodebench-historical-verified-trajectory",
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": str(row["question_content"]).strip()},
                    {"role": "assistant", "content": code.rstrip() + "\n"},
                ],
                "metadata": {
                    "task_id": f"livecodebench/{question_id}",
                    "question_id": question_id,
                    "contest_date": contest_date.isoformat(),
                    "difficulty": row.get("difficulty"),
                    "platform": row.get("platform"),
                    "verification": "full_livecodebench_harness_passed",
                    "source_policy": "preexisting_l20_system_selected_output",
                },
            }
        )

    admitted.sort(
        key=lambda row: stable_hash(f"{seed}:{row['metadata']['question_id']}")
    )
    selected = admitted[:max_records]
    if len(selected) < max_records:
        raise ValueError(f"only {len(selected)} verified LCB trajectories; need {max_records}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    report = {
        "eval_all_json": str(eval_all_json),
        "eval_all_sha256": sha256_file(eval_all_json),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "min_contest_date": min_contest_date,
        "max_contest_date": max_contest_date,
        "eligible_verified_records": len(admitted),
        "selected_records": len(selected),
        "seed": seed,
        "claim_boundary": (
            "These are training-only historical benchmark trajectories. They cannot support a "
            "claim on overlapping LiveCodeBench dates."
        ),
    }
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def compose_retention_sft_mixture(
    target_sft: Path,
    lcb_replay_sft: Path,
    mbpp_replay_sft: Path,
    output: Path,
    *,
    target_records: int = 414,
    lcb_records: int = 138,
    mbpp_records: int = 138,
    seed: int = 20260830,
) -> dict[str, Any]:
    """Create a deterministic 60/20/20 target/trajectory/function replay mix."""

    specifications = [
        ("rstar_target", target_sft, target_records),
        ("lcb_trajectory_replay", lcb_replay_sft, lcb_records),
        ("mbpp_function_replay", mbpp_replay_sft, mbpp_records),
    ]
    selected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    for source_name, path, count in specifications:
        rows = _load_jsonl(path)
        rows.sort(
            key=lambda row: stable_hash(
                f"{seed}:{source_name}:{_record_identity(row)}"
            )
        )
        if len(rows) < count:
            raise ValueError(f"{source_name} has {len(rows)} records; need {count}")
        for row in rows[:count]:
            metadata = dict(row.get("metadata") or {})
            metadata["mixture_source"] = source_name
            selected.append({**row, "metadata": metadata})
        source_counts[source_name] = count
        source_hashes[source_name] = sha256_file(path)

    selected.sort(
        key=lambda row: stable_hash(
            f"{seed}:mixture:{row['metadata']['mixture_source']}:{_record_identity(row)}"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    total = len(selected)
    report = {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "records": total,
        "source_counts": source_counts,
        "source_fractions": {name: count / total for name, count in source_counts.items()},
        "source_sha256": source_hashes,
        "seed": seed,
        "claim_boundary": (
            "Optimization mixture only. LCB records are historical training data and MBPP test "
            "or EvalPlus records are not included."
        ),
    }
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _record_identity(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("task_id"):
        return str(metadata["task_id"])
    return stable_hash(json.dumps(row.get("messages", []), sort_keys=True, ensure_ascii=False))


def _mbpp_user_prompt(description: str, tests: list[str]) -> str:
    assertions = "\n".join(tests)
    return (
        f"{description.strip()}\n\n"
        "Return a complete Python implementation satisfying these examples:\n"
        f"{assertions}"
    )
