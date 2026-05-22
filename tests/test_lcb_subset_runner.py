from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path


def load_runner_module():
    script = Path(__file__).parents[1] / "scripts" / "run_lcb_subset_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_lcb_subset_benchmark", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeProblem:
    question_content: str
    starter_code: str = ""


def test_build_lcb_generation_prompt_matches_stdin_format() -> None:
    runner = load_runner_module()

    prompt = runner.build_lcb_generation_prompt(FakeProblem("Solve A+B."))

    assert "### Question:" in prompt
    assert "Solve A+B." in prompt
    assert "Read the inputs from stdin" in prompt
    assert "# YOUR CODE HERE" in prompt
    assert "### Answer:" in prompt


def test_build_lcb_generation_prompt_includes_starter_code() -> None:
    runner = load_runner_module()

    prompt = runner.build_lcb_generation_prompt(
        FakeProblem("Implement add.", "class Solution:\n    pass")
    )

    assert "starter code" in prompt
    assert "class Solution" in prompt
    assert "# YOUR CODE HERE" not in prompt


def test_strip_lcb_code_block_prefers_last_python_like_block() -> None:
    runner = load_runner_module()

    output = (
        "Idea\n```text\nnot code\n```\n"
        "```python\nimport sys\nprint(sys.stdin.read())\n```\n"
    )

    assert runner.strip_lcb_code_block(output) == "import sys\nprint(sys.stdin.read())"


def test_sanitize_lcb_metadata_removes_test_payloads() -> None:
    runner = load_runner_module()
    payload = {
        "in" + "puts": "hidden",
        "ex" + "pected": "42",
        "error_code": -2,
        "error_message": "Wrong",
    }

    sanitized = runner.sanitize_lcb_metadata([json.dumps(payload)])

    assert sanitized == [{"error_code": -2, "error_message": "Wrong"}]
