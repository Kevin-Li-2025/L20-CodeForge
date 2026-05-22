from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from l20_codeforge.data.io import read_jsonl, write_jsonl
from l20_codeforge.data.real_datasets import RealTaskRecord
from l20_codeforge.data.sft import SYSTEM_PROMPT


class RealSFTRecord(BaseModel):
    messages: list[dict[str, str]]
    dataset: str
    instance_id: str
    repo: str
    reward_total: float = 1.0
    metadata: dict[str, object] = Field(default_factory=dict)


def real_task_to_sft_record(record: RealTaskRecord) -> RealSFTRecord:
    tests = []
    if record.fail_to_pass:
        tests.append("Failing tests to fix:")
        tests.extend(f"- {test}" for test in record.fail_to_pass)
    if record.pass_to_pass:
        tests.append("Regression tests to keep passing:")
        tests.extend(f"- {test}" for test in record.pass_to_pass)
    tests_text = "\n".join(tests) if tests else "No explicit test names provided."

    user_prompt = "\n".join(
        [
            f"Dataset: {record.dataset}",
            f"Instance ID: {record.instance_id}",
            f"Repository: {record.repo}",
            f"Base commit: {record.base_commit or 'not specified'}",
            "",
            "Issue:",
            record.problem_statement,
            "",
            tests_text,
            "",
            "Return only a unified diff patch.",
        ]
    )
    return RealSFTRecord(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": record.patch.strip() + "\n"},
        ],
        dataset=record.dataset,
        instance_id=record.instance_id,
        repo=record.repo,
        metadata={
            "split": record.split,
            "base_commit": record.base_commit,
            "issue_url": record.issue_url,
            "pr_url": record.pr_url,
            "fail_to_pass": record.fail_to_pass,
            "pass_to_pass": record.pass_to_pass,
            "language": record.language,
            "license": record.license,
        },
    )


def build_real_sft_jsonl(
    real_tasks_path: Path,
    output_path: Path,
    limit: int | None = None,
    min_patch_chars: int = 20,
) -> int:
    records = []
    for index, record in enumerate(read_jsonl(real_tasks_path, RealTaskRecord)):
        if limit is not None and index >= limit:
            break
        if len(record.patch.strip()) < min_patch_chars:
            continue
        records.append(real_task_to_sft_record(record))
    return write_jsonl(output_path, records)

