from __future__ import annotations

import ast
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

DEFAULT_TIMEOUT_SECONDS = 2.0
DEFAULT_MEMORY_LIMIT_MB = 512
DEFAULT_MAX_OUTPUT_CHARS = 100_000


class CodeTestCase(BaseModel):
    name: str | None = None
    stdin: str = Field(validation_alias=AliasChoices("stdin", "input"))
    expected_stdout: str = Field(
        validation_alias=AliasChoices("expected_stdout", "output", "expected_output")
    )


class CodeExecutionConfig(BaseModel):
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0)
    compile_timeout_seconds: float = Field(default=5.0, gt=0)
    memory_limit_mb: int = Field(default=DEFAULT_MEMORY_LIMIT_MB, gt=0)
    max_output_chars: int = Field(default=DEFAULT_MAX_OUTPUT_CHARS, ge=1024)
    comparison: Literal["tokens", "exact"] = "tokens"
    compile_weight: float = Field(default=0.1, ge=0)
    pass_fraction_weight: float = Field(default=0.7, ge=0)
    all_pass_weight: float = Field(default=0.2, ge=0)
    timeout_penalty: float = Field(default=0.2, ge=0)
    runtime_error_penalty: float = Field(default=0.1, ge=0)
    reject_unsafe_code: bool = True


class ProcessResult(BaseModel):
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    output_limited: bool = False


class CodeTestResult(BaseModel):
    index: int
    name: str | None = None
    status: Literal["passed", "wrong_answer", "runtime_error", "timeout", "output_limit"]
    passed: bool
    exit_code: int
    elapsed_seconds: float
    actual_stdout: str = ""
    expected_stdout: str = ""
    stderr: str = ""


class CodeExecutionReport(BaseModel):
    status: Literal["passed", "failed", "compile_error"]
    compiled: bool
    tests_passed: int
    tests_total: int
    pass_fraction: float
    all_passed: bool
    reward: float
    behavior_signature: str
    compile_result: ProcessResult
    test_results: list[CodeTestResult] = Field(default_factory=list)
    extracted_code: str
    elapsed_seconds: float


_FENCED_CODE = re.compile(
    r"```(?P<language>[A-Za-z0-9_+-]*)[^\n]*\n(?P<code>.*?)```",
    flags=re.DOTALL,
)


def extract_python_code(completion: str) -> str:
    """Extract a Python program while keeping raw-code completions unchanged."""

    matches = list(_FENCED_CODE.finditer(completion))
    if not matches:
        return completion.strip()

    python_blocks = [
        match.group("code").strip()
        for match in matches
        if match.group("language").lower() in {"python", "py"}
    ]
    if python_blocks:
        return python_blocks[-1]
    return max((match.group("code").strip() for match in matches), key=len, default="")


def evaluate_python_completion(
    completion: str,
    test_cases: list[CodeTestCase] | list[dict[str, str]],
    config: CodeExecutionConfig | None = None,
) -> CodeExecutionReport:
    """Compile and execute a standalone Python solution against stdin/stdout tests.

    Resource limits reduce accidental damage, but this subprocess runner is not a
    security boundary. Untrusted large-scale rollouts still require a container or
    another isolated worker.
    """

    active_config = config or CodeExecutionConfig()
    cases = [
        case if isinstance(case, CodeTestCase) else CodeTestCase.model_validate(case)
        for case in test_cases
    ]
    if not cases:
        raise ValueError("code execution reward requires at least one test case")

    code = extract_python_code(completion)
    started = time.monotonic()
    safety_violation = (
        find_python_safety_violation(code) if active_config.reject_unsafe_code else None
    )
    if safety_violation:
        return CodeExecutionReport(
            status="compile_error",
            compiled=False,
            tests_passed=0,
            tests_total=len(cases),
            pass_fraction=0.0,
            all_passed=False,
            reward=0.0,
            behavior_signature="S" + "-" * len(cases),
            compile_result=ProcessResult(
                exit_code=126,
                stderr=f"unsafe code rejected: {safety_violation}",
            ),
            extracted_code=code,
            elapsed_seconds=time.monotonic() - started,
        )
    with tempfile.TemporaryDirectory(prefix="l20-code-verifier-") as temp_dir:
        workdir = Path(temp_dir)
        solution = workdir / "solution.py"
        solution.write_text(code, encoding="utf-8")

        compile_result = _run_limited_process(
            [sys.executable, "-I", "-m", "py_compile", str(solution)],
            cwd=workdir,
            stdin="",
            config=active_config,
            timeout_seconds=active_config.compile_timeout_seconds,
        )
        if compile_result.exit_code != 0:
            return CodeExecutionReport(
                status="compile_error",
                compiled=False,
                tests_passed=0,
                tests_total=len(cases),
                pass_fraction=0.0,
                all_passed=False,
                reward=0.0,
                behavior_signature="C" + "-" * len(cases),
                compile_result=compile_result,
                extracted_code=code,
                elapsed_seconds=time.monotonic() - started,
            )

        results = [
            _run_test_case(
                solution=solution,
                case=case,
                index=index,
                workdir=workdir,
                config=active_config,
            )
            for index, case in enumerate(cases)
        ]

    passed = sum(result.passed for result in results)
    pass_fraction = passed / len(results)
    all_passed = passed == len(results)
    reward = (
        active_config.compile_weight
        + active_config.pass_fraction_weight * pass_fraction
        + active_config.all_pass_weight * int(all_passed)
    )
    if any(result.status == "timeout" for result in results):
        reward -= active_config.timeout_penalty
    if any(result.status in {"runtime_error", "output_limit"} for result in results):
        reward -= active_config.runtime_error_penalty
    reward = max(-1.0, min(1.0, reward))
    signature = "".join(
        {
            "passed": "P",
            "wrong_answer": "W",
            "runtime_error": "E",
            "timeout": "T",
            "output_limit": "O",
        }[result.status]
        for result in results
    )
    return CodeExecutionReport(
        status="passed" if all_passed else "failed",
        compiled=True,
        tests_passed=passed,
        tests_total=len(results),
        pass_fraction=pass_fraction,
        all_passed=all_passed,
        reward=reward,
        behavior_signature=signature,
        compile_result=compile_result,
        test_results=results,
        extracted_code=code,
        elapsed_seconds=time.monotonic() - started,
    )


_DENIED_IMPORT_ROOTS = {
    "builtins",
    "ctypes",
    "glob",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "tempfile",
}
_DENIED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "locals",
    "open",
    "setattr",
    "vars",
}


def find_python_safety_violation(code: str) -> str | None:
    """Reject common filesystem/process/network escape surfaces before execution.

    This is defense in depth, not a replacement for a container sandbox.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _DENIED_IMPORT_ROOTS:
                    return f"denied import {root}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in _DENIED_IMPORT_ROOTS:
                return f"denied import {root}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _DENIED_CALLS:
                return f"denied call {node.func.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"denied dunder attribute {node.attr}"
    return None


def _run_test_case(
    solution: Path,
    case: CodeTestCase,
    index: int,
    workdir: Path,
    config: CodeExecutionConfig,
) -> CodeTestResult:
    result = _run_limited_process(
        [sys.executable, "-I", str(solution)],
        cwd=workdir,
        stdin=case.stdin,
        config=config,
    )
    if result.timed_out:
        status = "timeout"
    elif result.output_limited:
        status = "output_limit"
    elif result.exit_code != 0:
        status = "runtime_error"
    elif _outputs_match(result.stdout, case.expected_stdout, config.comparison):
        status = "passed"
    else:
        status = "wrong_answer"
    return CodeTestResult(
        index=index,
        name=case.name,
        status=status,
        passed=status == "passed",
        exit_code=result.exit_code,
        elapsed_seconds=result.elapsed_seconds,
        actual_stdout=result.stdout,
        expected_stdout=case.expected_stdout,
        stderr=result.stderr,
    )


def _outputs_match(actual: str, expected: str, comparison: str) -> bool:
    if comparison == "tokens":
        return actual.split() == expected.split()
    return _normalize_exact(actual) == _normalize_exact(expected)


def _normalize_exact(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.replace("\r\n", "\n").splitlines()).strip()


def _run_limited_process(
    command: list[str],
    cwd: Path,
    stdin: str,
    config: CodeExecutionConfig,
    timeout_seconds: float | None = None,
) -> ProcessResult:
    started = time.monotonic()
    active_timeout = timeout_seconds or config.timeout_seconds
    stdout_path = cwd / f"stdout-{time.monotonic_ns()}.txt"
    stderr_path = cwd / f"stderr-{time.monotonic_ns()}.txt"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    timed_out = False
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            _limited_command(command, config, active_timeout),
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        try:
            process.communicate(
                input=stdin.encode("utf-8"),
                timeout=active_timeout,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    stdout = _read_limited(stdout_path, config.max_output_chars)
    stderr = _read_limited(stderr_path, config.max_output_chars)
    output_limited = (
        any(path.stat().st_size >= config.max_output_chars for path in (stdout_path, stderr_path))
        or process.returncode == -signal.SIGXFSZ
    )
    return ProcessResult(
        exit_code=124 if timed_out else int(process.returncode),
        stdout=stdout,
        stderr=stderr,
        elapsed_seconds=time.monotonic() - started,
        timed_out=timed_out,
        output_limited=output_limited,
    )


def _read_limited(path: Path, limit: int) -> str:
    with path.open("rb") as handle:
        return handle.read(limit).decode("utf-8", errors="replace")


def _limited_command(
    command: list[str], config: CodeExecutionConfig, timeout_seconds: float
) -> list[str]:
    if os.name != "posix":  # pragma: no cover - Windows compatibility
        return command

    limits = [
        ("RLIMIT_CPU", max(1, math.ceil(timeout_seconds) + 1)),
        ("RLIMIT_AS", config.memory_limit_mb * 1024 * 1024),
        ("RLIMIT_FSIZE", max(1024, config.max_output_chars)),
        ("RLIMIT_NOFILE", 64),
    ]
    launcher = (
        "import os, resource, sys\n"
        f"limits = {limits!r}\n"
        "for name, value in limits:\n"
        "    target = getattr(resource, name, None)\n"
        "    if target is None:\n"
        "        continue\n"
        "    try:\n"
        "        hard = resource.getrlimit(target)[1]\n"
        "        effective = value if hard < 0 else min(value, hard)\n"
        "        resource.setrlimit(target, (effective, effective))\n"
        "    except (OSError, ValueError):\n"
        "        pass\n"
        "os.execv(sys.argv[1], sys.argv[1:])\n"
    )
    return [sys.executable, "-I", "-c", launcher, *command]
