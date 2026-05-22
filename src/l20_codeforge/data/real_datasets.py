from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import json
import tempfile
import urllib.request
from typing import Any

from pydantic import BaseModel, Field

from l20_codeforge.data.io import write_json, write_jsonl


class RealDatasetSpec(BaseModel):
    key: str
    hf_id: str | None = None
    default_split: str
    description: str
    source_url: str
    task_type: str
    language: str
    executable: bool
    license_note: str
    priority: int


class RealTaskRecord(BaseModel):
    dataset: str
    split: str
    instance_id: str
    repo: str
    base_commit: str = ""
    problem_statement: str = ""
    patch: str = ""
    test_patch: str = ""
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    issue_url: str = ""
    pr_url: str = ""
    language: str = ""
    license: str = ""
    created_at: str | int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RealDatasetFetchReport(BaseModel):
    dataset: str
    hf_id: str | None
    split: str
    records: int
    output: str
    metadata_output: str
    source_url: str


REAL_DATASET_SPECS: dict[str, RealDatasetSpec] = {
    "swe-bench-lite": RealDatasetSpec(
        key="swe-bench-lite",
        hf_id="SWE-bench/SWE-bench_Lite",
        default_split="test",
        description="300 real GitHub issue/PR Python tasks for quick SWE-bench iteration.",
        source_url="https://www.swebench.com/SWE-bench/guides/datasets/",
        task_type="real_github_issue_pr",
        language="python",
        executable=True,
        license_note="Dataset hosted by SWE-bench; respect underlying repository licenses.",
        priority=1,
    ),
    "swe-bench-verified": RealDatasetSpec(
        key="swe-bench-verified",
        hf_id="SWE-bench/SWE-bench_Verified",
        default_split="test",
        description="500 expert-verified SWE-bench tasks intended as higher-quality evaluation.",
        source_url="https://www.swebench.com/SWE-bench/guides/datasets/",
        task_type="real_github_issue_pr",
        language="python",
        executable=True,
        license_note="Dataset hosted by SWE-bench; respect underlying repository licenses.",
        priority=2,
    ),
    "swe-gym": RealDatasetSpec(
        key="swe-gym",
        hf_id="SWE-Gym/SWE-Gym",
        default_split="train",
        description="2,438 real-world Python SWE training instances with executable environments.",
        source_url="https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
        task_type="real_github_issue_pr_training_env",
        language="python",
        executable=True,
        license_note="Check dataset card and underlying repository licenses.",
        priority=3,
    ),
    "swe-rebench-v2": RealDatasetSpec(
        key="swe-rebench-v2",
        hf_id="nebius/SWE-rebench-V2",
        default_split="train",
        description="32,079 multilingual real GitHub issue/PR tasks with Docker metadata.",
        source_url="https://huggingface.co/datasets/nebius/SWE-rebench-V2",
        task_type="real_github_issue_pr_multilingual",
        language="multilingual",
        executable=True,
        license_note="CC-BY-4.0 dataset metadata; respect per-repository license field.",
        priority=4,
    ),
    "pybughive": RealDatasetSpec(
        key="pybughive",
        hf_id=None,
        default_split="all",
        description="149 manually validated reproducible Python bugs from 11 projects.",
        source_url="https://pybughive.github.io/",
        task_type="real_python_bug_benchmark",
        language="python",
        executable=True,
        license_note="Use the project CLI/download; respect project and repository licenses.",
        priority=5,
    ),
    "bugsinpy": RealDatasetSpec(
        key="bugsinpy",
        hf_id=None,
        default_split="all",
        description="493 real Python bugs from 17 real-world Python programs.",
        source_url="https://arxiv.org/abs/2401.15481",
        task_type="real_python_bug_benchmark",
        language="python",
        executable=True,
        license_note="Use BugsInPy tooling; respect source project licenses.",
        priority=6,
    ),
    "bugsjs": RealDatasetSpec(
        key="bugsjs",
        hf_id=None,
        default_split="all",
        description="453 real JavaScript bugs from 10 mature Node.js projects.",
        source_url="https://bugsjs.github.io/",
        task_type="real_javascript_bug_benchmark",
        language="javascript",
        executable=True,
        license_note="Use BugsJS tooling; respect source project licenses.",
        priority=7,
    ),
}


def list_real_dataset_specs() -> list[RealDatasetSpec]:
    return sorted(REAL_DATASET_SPECS.values(), key=lambda item: item.priority)


def get_real_dataset_spec(key: str) -> RealDatasetSpec:
    try:
        return REAL_DATASET_SPECS[key]
    except KeyError as exc:
        known = ", ".join(sorted(REAL_DATASET_SPECS))
        raise ValueError(f"unknown real dataset {key!r}; known datasets: {known}") from exc


def fetch_hf_real_dataset(
    key: str,
    output_path: Path,
    split: str | None = None,
    limit: int | None = None,
    streaming: bool = False,
) -> RealDatasetFetchReport:
    spec = get_real_dataset_spec(key)
    if not spec.hf_id:
        raise ValueError(f"{key!r} does not have a Hugging Face dataset id; use source_url manually")
    dataset_split = split or spec.default_split

    raw_dataset = _load_hf_rows(spec.hf_id, dataset_split, streaming=streaming)
    records = []
    for row in _take(raw_dataset, limit):
        records.append(normalize_real_row(spec=spec, split=dataset_split, row=dict(row)))

    count = write_jsonl(output_path, records)
    metadata_output = output_path.with_suffix(".meta.json")
    report = RealDatasetFetchReport(
        dataset=spec.key,
        hf_id=spec.hf_id,
        split=dataset_split,
        records=count,
        output=str(output_path),
        metadata_output=str(metadata_output),
        source_url=spec.source_url,
    )
    write_json(metadata_output, {"spec": spec.model_dump(), "report": report.model_dump()})
    return report


def _load_hf_rows(hf_id: str, split: str, streaming: bool = False) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset

        return load_dataset(hf_id, split=split, streaming=streaming)
    except Exception as datasets_exc:
        if streaming:
            raise RuntimeError(
                f"datasets.load_dataset failed for {hf_id}:{split} and direct parquet fallback "
                f"does not support streaming: {datasets_exc}"
            ) from datasets_exc
        try:
            return list(_load_hf_parquet_rows(hf_id, split))
        except Exception as parquet_exc:
            raise RuntimeError(
                f"failed to fetch {hf_id}:{split} with datasets and direct parquet fallback. "
                f"datasets error: {datasets_exc}; parquet error: {parquet_exc}"
            ) from parquet_exc


def _load_hf_parquet_rows(hf_id: str, split: str) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("direct parquet fallback requires pyarrow") from exc

    tree_url = f"https://huggingface.co/api/datasets/{hf_id}/tree/main?recursive=1"
    with urllib.request.urlopen(tree_url, timeout=30) as response:
        tree = json.load(response)
    parquet_paths = [
        item["path"]
        for item in tree
        if item.get("type") == "file"
        and item.get("path", "").startswith(f"data/{split}-")
        and item.get("path", "").endswith(".parquet")
    ]
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet files found for split {split!r} in {hf_id}")

    with tempfile.TemporaryDirectory(prefix="l20-real-data-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for rel_path in parquet_paths:
            url = f"https://huggingface.co/datasets/{hf_id}/resolve/main/{rel_path}"
            local_path = tmp_root / rel_path.replace("/", "__")
            urllib.request.urlretrieve(url, local_path)
            table = pq.read_table(local_path)
            for row in table.to_pylist():
                yield row


def normalize_real_row(spec: RealDatasetSpec, split: str, row: dict[str, Any]) -> RealTaskRecord:
    instance_id = str(_first_present(row, "instance_id", "id", "task_id") or "")
    repo = str(_first_present(row, "repo", "repository", "repo_name") or "")
    metadata_keys = set(row) - {
        "instance_id",
        "id",
        "task_id",
        "repo",
        "repository",
        "repo_name",
        "base_commit",
        "problem_statement",
        "patch",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "fail_to_pass",
        "pass_to_pass",
        "issue_url",
        "pr_url",
        "language",
        "license",
        "created_at",
    }
    return RealTaskRecord(
        dataset=spec.key,
        split=split,
        instance_id=instance_id,
        repo=repo,
        base_commit=str(row.get("base_commit") or ""),
        problem_statement=str(_first_present(row, "problem_statement", "issue", "task") or ""),
        patch=str(row.get("patch") or ""),
        test_patch=str(row.get("test_patch") or ""),
        fail_to_pass=_list_of_str(_first_present(row, "FAIL_TO_PASS", "fail_to_pass")),
        pass_to_pass=_list_of_str(_first_present(row, "PASS_TO_PASS", "pass_to_pass")),
        issue_url=str(row.get("issue_url") or ""),
        pr_url=str(row.get("pr_url") or ""),
        language=str(row.get("language") or spec.language),
        license=str(row.get("license") or ""),
        created_at=row.get("created_at"),
        metadata={key: row[key] for key in sorted(metadata_keys)},
    )


def _take(dataset: Iterable[Any], limit: int | None) -> Iterable[Any]:
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield row


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]
