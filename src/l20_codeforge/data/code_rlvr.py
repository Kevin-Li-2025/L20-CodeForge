from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

RSTAR_DATASET = "microsoft/rStar-Coder"
RSTAR_CONFIG = "synthetic_rl_testcase"
RSTAR_LICENSE = "CC-BY-4.0"
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Return only a complete Python 3 "
    "program that reads from standard input and writes to standard output."
)


def materialize_rstar_code_rlvr(
    output_dir: Path,
    *,
    train_tasks: int = 800,
    dev_tasks: int = 200,
    retention_tasks: int = 0,
    final_tasks: int = 0,
    seed: int = 20260829,
    min_tests: int = 8,
    max_tests: int = 24,
    max_case_input_chars: int = 131_072,
    max_case_output_chars: int = 65_536,
    max_prompt_chars: int = 24_000,
    near_duplicate_threshold: float = 0.85,
    cache_dir: Path | None = None,
    parquet_urls: Sequence[str] | None = None,
    use_rows_api: bool = False,
    rows_api_limit: int = 4000,
    rows_api_batch_size: int = 20,
    rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Freeze disjoint rStar train/dev JSONL with deterministic test subsampling.

    By default this uses the Hugging Face Dataset Server's converted parquet
    surface. That surface currently exposes a 4,000-row partial conversion and
    avoids downloading the hundreds-of-gigabytes original testcase directory.
    Tests may inject ``rows`` to exercise the split/filter logic without network.
    """

    if train_tasks <= 0 or dev_tasks <= 0:
        raise ValueError("train_tasks and dev_tasks must be positive")
    if retention_tasks < 0 or final_tasks < 0:
        raise ValueError("retention_tasks and final_tasks must be non-negative")
    if max_tests < min_tests:
        raise ValueError("max_tests must be at least min_tests")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_urls = list(parquet_urls or [])
    if rows is None:
        if use_rows_api:
            rows = iter_dataset_server_rows(
                RSTAR_DATASET,
                RSTAR_CONFIG,
                limit=rows_api_limit,
                batch_size=rows_api_batch_size,
            )
        else:
            if not source_urls:
                source_urls = discover_dataset_server_parquet_urls(RSTAR_DATASET, RSTAR_CONFIG)
            rows = iter_parquet_rows(source_urls, cache_dir=cache_dir)

    candidates: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    seen_prompts: set[str] = set()
    rows_seen = 0
    for raw in rows:
        rows_seen += 1
        task, rejection = rstar_row_to_code_task(
            raw,
            seed=seed,
            min_tests=min_tests,
            max_tests=max_tests,
            max_case_input_chars=max_case_input_chars,
            max_case_output_chars=max_case_output_chars,
            max_prompt_chars=max_prompt_chars,
        )
        if task is None:
            rejection_counts[rejection] = rejection_counts.get(rejection, 0) + 1
            continue
        prompt_hash = task["prompt_sha256"]
        if prompt_hash in seen_prompts:
            rejection_counts["exact_duplicate_prompt"] = (
                rejection_counts.get("exact_duplicate_prompt", 0) + 1
            )
            continue
        seen_prompts.add(prompt_hash)
        candidates.append(task)

    candidates.sort(key=lambda row: stable_hash(f"{seed}:split:{row['task_id']}"))
    dev = candidates[:dev_tasks]
    dev_shingles = [prompt_shingles(row["prompt"]) for row in dev]
    train: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    near_duplicates: list[dict[str, Any]] = []
    for task in candidates[dev_tasks:]:
        shingles = prompt_shingles(task["prompt"])
        max_similarity = max(
            (jaccard_similarity(shingles, dev_tokens) for dev_tokens in dev_shingles),
            default=0.0,
        )
        if max_similarity >= near_duplicate_threshold:
            near_duplicates.append(
                {"task_id": task["task_id"], "max_dev_jaccard": round(max_similarity, 6)}
            )
            continue
        if len(train) < train_tasks:
            train.append(task)
        else:
            remaining.append(task)

    if len(dev) < dev_tasks or len(train) < train_tasks:
        raise ValueError(
            "insufficient admitted tasks after filtering: "
            f"train={len(train)}/{train_tasks}, dev={len(dev)}/{dev_tasks}"
        )

    retention, final, holdout_near_duplicates = _select_disjoint_holdouts(
        remaining,
        prior=dev + train,
        retention_tasks=retention_tasks,
        final_tasks=final_tasks,
        near_duplicate_threshold=near_duplicate_threshold,
    )
    if len(retention) < retention_tasks or len(final) < final_tasks:
        raise ValueError(
            "insufficient holdout tasks after filtering: "
            f"retention={len(retention)}/{retention_tasks}, final={len(final)}/{final_tasks}"
        )

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    retention_path = output_dir / "retention.jsonl"
    final_path = output_dir / "final.jsonl"
    if retention_tasks:
        write_jsonl(retention_path, retention)
    if final_tasks:
        write_jsonl(final_path, final)

    manifest = {
        "dataset": RSTAR_DATASET,
        "config": RSTAR_CONFIG,
        "license": RSTAR_LICENSE,
        "claim_boundary": (
            "The source dataset card describes a verified coding corpus. This receipt "
            "freezes a development/training split; it is not model-quality evidence."
        ),
        "dataset_server_partial_conversion": True,
        "source_parquet_urls": source_urls,
        "source_rows_api": use_rows_api,
        "rows_api_limit": rows_api_limit if use_rows_api else None,
        "rows_api_batch_size": rows_api_batch_size if use_rows_api else None,
        "rows_seen": rows_seen,
        "candidate_tasks": len(candidates),
        "train_tasks": len(train),
        "dev_tasks": len(dev),
        "retention_tasks": len(retention),
        "final_tasks": len(final),
        "seed": seed,
        "min_tests": min_tests,
        "max_tests": max_tests,
        "max_case_input_chars": max_case_input_chars,
        "max_case_output_chars": max_case_output_chars,
        "max_prompt_chars": max_prompt_chars,
        "near_duplicate_threshold": near_duplicate_threshold,
        "near_duplicates_removed": near_duplicates,
        "holdout_near_duplicates_removed": holdout_near_duplicates,
        "rejection_counts": rejection_counts,
        "train_task_ids": [row["task_id"] for row in train],
        "dev_task_ids": [row["task_id"] for row in dev],
        "retention_task_ids": [row["task_id"] for row in retention],
        "final_task_ids": [row["task_id"] for row in final],
        "train_jsonl": str(train_path),
        "dev_jsonl": str(dev_path),
        "train_sha256": sha256_file(train_path),
        "dev_sha256": sha256_file(dev_path),
        "retention_jsonl": str(retention_path) if retention_tasks else None,
        "final_jsonl": str(final_path) if final_tasks else None,
        "retention_sha256": sha256_file(retention_path) if retention_tasks else None,
        "final_sha256": sha256_file(final_path) if final_tasks else None,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _select_disjoint_holdouts(
    candidates: Sequence[dict[str, Any]],
    *,
    prior: Sequence[dict[str, Any]],
    retention_tasks: int,
    final_tasks: int,
    near_duplicate_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select holdouts without changing the historical train/dev assignment."""

    if retention_tasks == 0 and final_tasks == 0:
        return [], [], []
    selected_context = [(row["task_id"], prompt_shingles(row["prompt"])) for row in prior]
    retention: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for task in candidates:
        shingles = prompt_shingles(task["prompt"])
        nearest_id = None
        max_similarity = 0.0
        for task_id, existing in selected_context:
            similarity = jaccard_similarity(shingles, existing)
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_id = task_id
        if max_similarity >= near_duplicate_threshold:
            rejected.append(
                {
                    "task_id": task["task_id"],
                    "nearest_task_id": nearest_id,
                    "max_jaccard": round(max_similarity, 6),
                }
            )
            continue
        if len(retention) < retention_tasks:
            retention.append(task)
        elif len(final) < final_tasks:
            final.append(task)
        else:
            break
        selected_context.append((task["task_id"], shingles))
    return retention, final, rejected


def discover_dataset_server_parquet_urls(dataset: str, config: str) -> list[str]:
    query = urllib.parse.urlencode({"dataset": dataset})
    url = f"https://datasets-server.huggingface.co/parquet?{query}"
    with urllib.request.urlopen(url, timeout=90) as response:
        payload = json.load(response)
    urls = [
        item["url"]
        for item in payload.get("parquet_files", [])
        if item.get("config") == config and item.get("split") == "train"
    ]
    if not urls:
        raise RuntimeError(f"no converted parquet URLs found for {dataset}/{config}")
    return urls


def iter_parquet_rows(urls: Sequence[str], cache_dir: Path | None) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files={"train": list(urls)},
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    for row in dataset:
        yield dict(row)


def iter_dataset_server_rows(
    dataset: str,
    config: str,
    *,
    limit: int,
    batch_size: int = 20,
    retries: int = 5,
) -> Iterable[dict[str, Any]]:
    """Iterate Dataset Server rows without requiring pyarrow or full parquet downloads."""

    if limit <= 0 or batch_size <= 0:
        raise ValueError("limit and batch_size must be positive")
    fetched = 0
    while fetched < limit:
        length = min(batch_size, limit - fetched)
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": "train",
                "offset": fetched,
                "length": length,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=180) as response:
                    payload = json.load(response)
                break
            except Exception as exc:  # pragma: no cover - exercised by live retries
                last_error = exc
                if attempt + 1 >= retries:
                    raise RuntimeError(f"failed Dataset Server batch offset={fetched}") from exc
                time.sleep(min(2**attempt, 10))
        else:  # pragma: no cover - loop always breaks or raises
            raise RuntimeError(f"failed Dataset Server batch offset={fetched}") from last_error
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            yield dict(item["row"])
        fetched += len(rows)
        if fetched % 100 == 0 or fetched >= limit:
            print(f"Dataset Server rows fetched: {fetched}/{limit}", flush=True)
        if len(rows) < length:
            break


def rstar_row_to_code_task(
    row: dict[str, Any],
    *,
    seed: int,
    min_tests: int,
    max_tests: int,
    max_case_input_chars: int,
    max_case_output_chars: int,
    max_prompt_chars: int,
) -> tuple[dict[str, Any] | None, str]:
    task_id = str(row.get("question_id") or "").strip()
    prompt = str(row.get("question") or "").strip()
    if not task_id or not prompt:
        return None, "missing_id_or_prompt"
    if len(prompt) > max_prompt_chars:
        return None, "prompt_too_long"

    try:
        inputs = json.loads(str(row.get("inputs") or "[]"))
        outputs = json.loads(str(row.get("outputs") or "[]"))
    except json.JSONDecodeError:
        return None, "invalid_test_json"
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        return None, "misaligned_tests"

    unique: dict[str, dict[str, str]] = {}
    for stdin, stdout in zip(inputs, outputs, strict=True):
        stdin_text = str(stdin)
        stdout_text = str(stdout)
        if len(stdin_text) > max_case_input_chars:
            continue
        if len(stdout_text) > max_case_output_chars:
            continue
        key = stable_hash(stdin_text + "\0" + stdout_text)
        unique.setdefault(key, {"stdin": stdin_text, "expected_stdout": stdout_text})
    if len(unique) < min_tests:
        return None, "too_few_bounded_tests"

    tests = sorted(
        unique.values(),
        key=lambda case: stable_hash(
            f"{seed}:test:{task_id}:" + case["stdin"] + "\0" + case["expected_stdout"]
        ),
    )[:max_tests]
    normalized_prompt = normalize_prompt(prompt)
    record = {
        "task_id": task_id,
        "prompt": prompt,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "tests": tests,
        "prompt_sha256": stable_hash(normalized_prompt),
        "tests_sha256": stable_hash(json.dumps(tests, sort_keys=True, ensure_ascii=False)),
        "source": {"dataset": RSTAR_DATASET, "config": RSTAR_CONFIG, "license": RSTAR_LICENSE},
    }
    return record, "admitted"


def prompt_shingles(text: str, width: int = 5) -> set[str]:
    tokens = re.findall(r"[a-z0-9_]+", normalize_prompt(text))
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
