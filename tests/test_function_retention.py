from __future__ import annotations

from l20_codeforge.evals.function_retention import evaluate_function_assertions
from l20_codeforge.rewards.code_execution import CodeExecutionConfig


def test_function_assertions_pass_and_fail() -> None:
    config = CodeExecutionConfig(timeout_seconds=1.0, compile_timeout_seconds=2.0)
    passed = evaluate_function_assertions(
        "def square(x):\n    return x * x",
        setup_code="",
        tests=["assert square(3) == 9", "assert square(-2) == 4"],
        config=config,
    )
    failed = evaluate_function_assertions(
        "def square(x):\n    return x",
        setup_code="",
        tests=["assert square(3) == 9"],
        config=config,
    )

    assert passed["status"] == "passed"
    assert passed["all_passed"] is True
    assert failed["status"] == "failed"
    assert failed["all_passed"] is False


def test_function_assertions_fail_closed_on_unsafe_code() -> None:
    report = evaluate_function_assertions(
        "import os\ndef answer():\n    return os.getcwd()",
        setup_code="",
        tests=["assert answer()"],
    )

    assert report["status"] == "unsafe"
    assert report["compiled"] is False
    assert report["exit_code"] == 126
