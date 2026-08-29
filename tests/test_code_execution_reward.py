from __future__ import annotations

from l20_codeforge.rewards.code_execution import (
    CodeExecutionConfig,
    CodeTestCase,
    evaluate_python_completion,
    extract_python_code,
    find_python_safety_violation,
)
from l20_codeforge.training.reward_functions import (
    code_binary_execution_reward,
    code_execution_reward,
)

CASES = [
    CodeTestCase(name="small", input="2\n", output="4\n"),
    CodeTestCase(name="edge", input="-3\n", output="9\n"),
]


def test_extract_python_code_prefers_last_python_fence() -> None:
    completion = """Reasoning.
```python
print('draft')
```
Final:
```python
x = int(input())
print(x * x)
```
"""

    assert extract_python_code(completion).startswith("x = int(input())")


def test_code_execution_reward_passes_all_tests() -> None:
    report = evaluate_python_completion(
        "x = int(input())\nprint(x * x)\n",
        CASES,
    )

    assert report.status == "passed"
    assert report.compiled is True
    assert report.tests_passed == 2
    assert report.pass_fraction == 1.0
    assert report.reward == 1.0
    assert report.behavior_signature == "PP"


def test_code_execution_reward_exposes_partial_and_compile_failures() -> None:
    partial = evaluate_python_completion(
        "x = int(input())\nprint(x + 2)\n",
        CASES,
    )
    compile_error = evaluate_python_completion("def broken(:\n", CASES)

    assert partial.status == "failed"
    assert partial.tests_passed == 1
    assert partial.pass_fraction == 0.5
    assert 0.1 < partial.reward < 1.0
    assert partial.behavior_signature == "PW"
    assert compile_error.status == "compile_error"
    assert compile_error.reward == 0.0
    assert compile_error.behavior_signature == "C--"


def test_code_execution_reward_marks_timeout() -> None:
    report = evaluate_python_completion(
        "while True:\n    pass\n",
        [CodeTestCase(input="", output="")],
        CodeExecutionConfig(timeout_seconds=0.1),
    )

    assert report.test_results[0].status == "timeout"
    assert report.behavior_signature == "T"
    assert report.reward <= 0.1


def test_parallel_test_workers_preserve_order_and_result() -> None:
    serial = evaluate_python_completion(
        "x = int(input())\nprint(x * x)",
        CASES,
        CodeExecutionConfig(workers=1),
    )
    parallel = evaluate_python_completion(
        "x = int(input())\nprint(x * x)",
        CASES,
        CodeExecutionConfig(workers=2),
    )
    assert parallel.behavior_signature == serial.behavior_signature == "PP"
    assert [result.index for result in parallel.test_results] == [0, 1]
    assert parallel.reward == serial.reward == 1.0


def test_rejects_common_filesystem_and_process_escape_surfaces() -> None:
    assert find_python_safety_violation("import os\nprint(os.getcwd())") == "denied import os"
    assert find_python_safety_violation("open('/tmp/x', 'w')") == "denied call open"
    report = evaluate_python_completion(
        "import subprocess\nsubprocess.run(['echo', 'x'])",
        [{"input": "", "output": "x"}],
    )
    assert report.compiled is False
    assert report.behavior_signature.startswith("S")
    assert report.compile_result.exit_code == 126


def test_trl_reward_helpers_support_dense_and_binary_ablation() -> None:
    completions = [
        [{"role": "assistant", "content": "x = int(input())\nprint(x * x)"}],
        [{"role": "assistant", "content": "print(0)"}],
    ]
    tests = [[case.model_dump() for case in CASES]]

    dense = code_execution_reward(completions, tests=tests)
    binary = code_binary_execution_reward(completions, tests=tests)

    assert dense[0] == 1.0
    assert dense[1] < dense[0]
    assert binary == [1.0, 0.0]
