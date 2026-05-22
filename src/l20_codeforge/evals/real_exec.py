from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from l20_codeforge.data.io import read_jsonl, write_json
from l20_codeforge.data.real_datasets import RealTaskRecord


DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_OUTPUT_CHARS = 20000
TIMEOUT_EXIT_CODE = 124


class CommandResult(BaseModel):
    name: str
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0
    timed_out: bool = False


class PatchAttemptResult(BaseModel):
    name: str
    status: str
    setup: list[CommandResult] = Field(default_factory=list)
    apply_results: list[CommandResult] = Field(default_factory=list)
    test: CommandResult | None = None
    patch_chars: int = 0


class RealPatchEvalReport(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    worktree: str
    test_command: str
    candidate_name: str
    candidate_patch_file: str
    verdict: str
    notes: list[str] = Field(default_factory=list)
    base_result: PatchAttemptResult
    gold_result: PatchAttemptResult
    candidate_result: PatchAttemptResult


def load_real_task(path: Path, instance_id: str) -> RealTaskRecord:
    for record in read_jsonl(path, RealTaskRecord):
        if record.instance_id == instance_id:
            return record
    raise ValueError(f"instance_id {instance_id!r} was not found in {path}")


def parse_swebench_tests(values: Iterable[Any] | Any) -> list[str]:
    """Flatten SWE-bench FAIL_TO_PASS/PASS_TO_PASS fields.

    Some dataset exports encode the test list as a single JSON string, e.g.
    ``['["test_a", "test_b"]']``. The executable harness needs the actual labels.
    """

    flattened: list[str] = []
    for value in _as_iterable(values):
        flattened.extend(_flatten_test_value(value))
    return [item for item in flattened if item]


def normalize_unittest_label(label: str) -> str:
    """Convert unittest verbose labels into Django/SWE-bench runnable labels."""

    stripped = label.strip()
    match = re.match(r"^(?P<method>[^()\s]+)\s+\((?P<classpath>[^)]+)\)$", stripped)
    if match:
        return f"{match.group('classpath')}.{match.group('method')}"
    return stripped


def infer_test_command(record: RealTaskRecord, test_command: str | None = None) -> str:
    if test_command:
        return test_command

    tests = parse_swebench_tests(record.fail_to_pass) or parse_swebench_tests(record.pass_to_pass)
    if not tests:
        raise ValueError(
            f"{record.instance_id} does not include fail_to_pass/pass_to_pass tests; "
            "pass --test-command explicitly"
        )

    test_labels = [normalize_unittest_label(test) for test in tests]
    quoted_labels = " ".join(shlex.quote(test_label) for test_label in test_labels)
    if record.repo == "django/django":
        return f"PYTHONPATH=. python tests/runtests.py {quoted_labels} --verbosity=2 --noinput"
    return f"python -m pytest {quoted_labels} -q"


def evaluate_real_patch(
    real_tasks_path: Path,
    instance_id: str,
    patch_file: Path,
    output: Path,
    repos_dir: Path = Path("artifacts/real_eval/repos"),
    repo_dir: Path | None = None,
    test_command: str | None = None,
    candidate_name: str = "candidate",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    fetch_existing: bool = True,
    max_output_chars: int = DEFAULT_OUTPUT_CHARS,
    keep_worktree_state: bool = False,
) -> RealPatchEvalReport:
    record = load_real_task(real_tasks_path, instance_id)
    candidate_patch = patch_file.read_text(encoding="utf-8")
    command = infer_test_command(record, test_command)
    worktree = repo_dir or ensure_repo_clone(
        record.repo,
        repos_dir=repos_dir,
        fetch_existing=fetch_existing,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )

    base_result = run_patch_attempt(
        record=record,
        repo_dir=worktree,
        name="base",
        patches=[],
        test_command=command,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )
    gold_result = run_patch_attempt(
        record=record,
        repo_dir=worktree,
        name="gold",
        patches=[("gold_patch", record.patch)],
        test_command=command,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )
    candidate_result = run_patch_attempt(
        record=record,
        repo_dir=worktree,
        name=candidate_name,
        patches=[("candidate_patch", candidate_patch)],
        test_command=command,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )

    verdict, notes = classify_report(base_result, gold_result, candidate_result)
    if not keep_worktree_state:
        try:
            cleanup = reset_to_base(
                repo_dir=worktree,
                base_commit=record.base_commit,
                timeout_seconds=timeout_seconds,
                max_output_chars=max_output_chars,
            )
        except Exception as exc:  # pragma: no cover - defensive cleanup path
            notes.append(f"cleanup_failed: {exc}")
        else:
            if _has_failed(cleanup):
                notes.append("cleanup_failed")

    report = RealPatchEvalReport(
        instance_id=record.instance_id,
        repo=record.repo,
        base_commit=record.base_commit,
        worktree=str(worktree),
        test_command=command,
        candidate_name=candidate_name,
        candidate_patch_file=str(patch_file),
        verdict=verdict,
        notes=notes,
        base_result=base_result,
        gold_result=gold_result,
        candidate_result=candidate_result,
    )
    write_json(output, report)
    return report


def ensure_repo_clone(
    repo: str,
    repos_dir: Path,
    fetch_existing: bool = True,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_OUTPUT_CHARS,
) -> Path:
    repos_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = repos_dir / repo.replace("/", "__")
    url = f"https://github.com/{repo}.git"
    if not repo_dir.exists():
        result = _run_command(
            ["git", "clone", "--filter=blob:none", url, str(repo_dir)],
            cwd=repos_dir,
            name="clone_repo",
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"failed to clone {url}: {result.stderr or result.stdout}")
    elif fetch_existing:
        result = _run_command(
            ["git", "fetch", "--all", "--tags", "--quiet"],
            cwd=repo_dir,
            name="fetch_repo",
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"failed to fetch {repo_dir}: {result.stderr or result.stdout}")
    return repo_dir


def run_patch_attempt(
    record: RealTaskRecord,
    repo_dir: Path,
    name: str,
    patches: list[tuple[str, str]],
    test_command: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_OUTPUT_CHARS,
) -> PatchAttemptResult:
    setup = reset_to_base(
        repo_dir=repo_dir,
        base_commit=record.base_commit,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )
    apply_results: list[CommandResult] = []

    if record.test_patch:
        apply_results.append(
            apply_patch_text(
                repo_dir=repo_dir,
                patch=record.test_patch,
                name="apply_test_patch",
                timeout_seconds=timeout_seconds,
                max_output_chars=max_output_chars,
            )
        )

    for patch_name, patch_text in patches:
        if _has_failed(apply_results):
            break
        apply_results.append(
            apply_patch_text(
                repo_dir=repo_dir,
                patch=patch_text,
                name=f"apply_{patch_name}",
                timeout_seconds=timeout_seconds,
                max_output_chars=max_output_chars,
            )
        )

    patch_chars = sum(len(patch_text) for _, patch_text in patches)
    if _has_failed(setup):
        return PatchAttemptResult(
            name=name,
            status="setup_failed",
            setup=setup,
            apply_results=apply_results,
            patch_chars=patch_chars,
        )
    if _has_failed(apply_results):
        return PatchAttemptResult(
            name=name,
            status="apply_failed",
            setup=setup,
            apply_results=apply_results,
            patch_chars=patch_chars,
        )

    test = _run_shell(
        test_command,
        cwd=repo_dir,
        name="run_test",
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )
    status = "passed" if test.exit_code == 0 else "failed"
    return PatchAttemptResult(
        name=name,
        status=status,
        setup=setup,
        apply_results=apply_results,
        test=test,
        patch_chars=patch_chars,
    )


def reset_to_base(
    repo_dir: Path,
    base_commit: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_OUTPUT_CHARS,
) -> list[CommandResult]:
    if not (repo_dir / ".git").exists():
        raise ValueError(f"{repo_dir} is not a git repository")
    if not base_commit:
        raise ValueError("record is missing base_commit")

    results = [
        _run_command(
            ["git", "checkout", "--quiet", base_commit],
            cwd=repo_dir,
            name="checkout_base",
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        ),
        _run_command(
            ["git", "reset", "--hard", "--quiet", base_commit],
            cwd=repo_dir,
            name="reset_base",
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        ),
        _run_command(
            ["git", "clean", "-fdx", "--quiet"],
            cwd=repo_dir,
            name="clean_repo",
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        ),
    ]
    return results


def apply_patch_text(
    repo_dir: Path,
    patch: str,
    name: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_OUTPUT_CHARS,
) -> CommandResult:
    return _run_command(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=repo_dir,
        name=name,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
        input_text=patch,
    )


def classify_report(
    base_result: PatchAttemptResult,
    gold_result: PatchAttemptResult,
    candidate_result: PatchAttemptResult,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    if base_result.status == "passed":
        notes.append("base_passed_with_test_patch")
    if gold_result.status != "passed":
        notes.append("gold_did_not_pass")
    if base_result.status == "passed" or gold_result.status != "passed":
        return "harness_invalid", notes
    if candidate_result.status == "passed":
        return "resolved", notes
    if candidate_result.status == "apply_failed":
        return "candidate_apply_failed", notes
    if candidate_result.status == "setup_failed":
        return "candidate_setup_failed", notes
    return "candidate_failed", notes


def _as_iterable(values: Iterable[Any] | Any) -> Iterable[Any]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    try:
        return iter(values)
    except TypeError:
        return [values]


def _flatten_test_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            items.extend(_flatten_test_value(item))
        return items
    raw = str(value).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    if isinstance(parsed, (list, tuple)):
        return [item for nested in parsed for item in _flatten_test_value(nested)]
    if isinstance(parsed, str):
        return [parsed] if parsed else []
    return [str(parsed)]


def _has_failed(results: list[CommandResult]) -> bool:
    return any(result.exit_code != 0 for result in results)


def _run_shell(
    command: str,
    cwd: Path,
    name: str,
    timeout_seconds: int,
    max_output_chars: int,
) -> CommandResult:
    return _run_command(
        command,
        cwd=cwd,
        name=name,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
        shell=True,
    )


def _run_command(
    command: list[str] | str,
    cwd: Path,
    name: str,
    timeout_seconds: int,
    max_output_chars: int,
    input_text: str | None = None,
    shell: bool = False,
) -> CommandResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            shell=shell,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        exit_code = TIMEOUT_EXIT_CODE
        stdout = _coerce_output(exc.stdout)
        stderr = _coerce_output(exc.stderr)
        timed_out = True

    elapsed = time.monotonic() - started
    command_text = command if isinstance(command, str) else shlex.join(command)
    return CommandResult(
        name=name,
        command=command_text,
        exit_code=exit_code,
        stdout=_tail(stdout, max_output_chars),
        stderr=_tail(stderr, max_output_chars),
        elapsed_seconds=round(elapsed, 3),
        timed_out=timed_out,
    )


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _tail(value: str | None, max_chars: int) -> str:
    if not value:
        return ""
    return value[-max_chars:]
