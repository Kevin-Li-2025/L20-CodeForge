from __future__ import annotations

from pathlib import Path

from l20_codeforge.evals.evalplus_runner import (
    build_evalplus_prompt,
    count_existing_samples,
    parse_evalplus_pass_at_1,
    parse_evalplus_scores,
    select_evalplus_tasks,
    select_evalplus_by_base_tests,
    strip_markdown_code_fence,
)


def test_select_evalplus_tasks_respects_id_range_and_limit() -> None:
    tasks = {f"HumanEval/{idx}": {"prompt": str(idx)} for idx in range(10)}

    selected = select_evalplus_tasks(tasks, limit=3, id_start=2, id_end=8)

    assert [task_id for task_id, _ in selected] == ["HumanEval/2", "HumanEval/3", "HumanEval/4"]


def test_strip_markdown_code_fence_prefers_python_block() -> None:
    text = "Here is code:\n```python\ndef f():\n    return 1\n```\n"

    assert strip_markdown_code_fence(text) == "def f():\n    return 1"


def test_build_evalplus_prompt_requests_code_only() -> None:
    prompt = build_evalplus_prompt("def add(a, b):\n    pass")

    assert "Return only valid Python code" in prompt
    assert "def add" in prompt


def test_count_existing_samples(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        '{"task_id": "HumanEval/0", "solution": "a"}\n'
        '{"task_id": "HumanEval/0", "solution": "b"}\n'
        '{"task_id": "HumanEval/1", "solution": "c"}\n',
        encoding="utf-8",
    )

    assert count_existing_samples(samples) == {"HumanEval/0": 2, "HumanEval/1": 1}


def test_parse_evalplus_pass_at_1() -> None:
    stdout = """humaneval (base tests)
pass@1:
\t0.890
humaneval+ (base + extra tests)
pass@1:
\t0.848
"""

    assert parse_evalplus_pass_at_1(stdout) == {"base": 0.89, "plus": 0.848}


def test_parse_evalplus_pass_at_1_inline_format() -> None:
    stdout = """humaneval (base tests)
pass@1:\t0.415
humaneval+ (base + extra tests)
pass@1:\t0.409
"""

    assert parse_evalplus_pass_at_1(stdout) == {"base": 0.415, "plus": 0.409}


def test_parse_evalplus_scores_multiple_k() -> None:
    stdout = """humaneval (base tests)
pass@1:\t0.890
pass@10:\t0.970
humaneval+ (base + extra tests)
pass@1:\t0.848
pass@10:\t0.951
"""

    assert parse_evalplus_scores(stdout) == {
        "base_pass@1": 0.89,
        "base_pass@10": 0.97,
        "plus_pass@1": 0.848,
        "plus_pass@10": 0.951,
    }


def test_select_evalplus_by_base_tests(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        '{"task_id": "HumanEval/0", "solution": "bad"}\n'
        '{"task_id": "HumanEval/0", "solution": "good"}\n'
        '{"task_id": "HumanEval/1", "solution": "fallback"}\n',
        encoding="utf-8",
    )
    eval_results = tmp_path / "results.json"
    eval_results.write_text(
        """
{
  "eval": {
    "HumanEval/0": [
      {"base_status": "fail", "plus_status": "fail"},
      {"base_status": "pass", "plus_status": "pass"}
    ],
    "HumanEval/1": [
      {"base_status": "fail", "plus_status": "fail"}
    ]
  }
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "selected.jsonl"

    report = select_evalplus_by_base_tests(samples, eval_results, output)

    assert report.selected_base_pass == 1
    assert report.selected_plus_pass == 1
    assert report.fallback_tasks == ["HumanEval/1"]
    assert output.read_text(encoding="utf-8").splitlines() == [
        '{"task_id": "HumanEval/0", "solution": "good"}',
        '{"task_id": "HumanEval/1", "solution": "fallback"}',
    ]
