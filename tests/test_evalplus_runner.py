from __future__ import annotations

import json
from pathlib import Path

from l20_codeforge.evals.evalplus_runner import (
    build_evalplus_prompt,
    choose_evalplus_candidate_index,
    count_existing_samples,
    parse_evalplus_pass_at_1,
    parse_evalplus_scores,
    run_prompt_doctests,
    select_evalplus_tasks,
    select_evalplus_by_base_tests,
    select_evalplus_by_prompt_doctests_from_tasks,
    strip_markdown_code_fence,
)


def test_select_evalplus_tasks_respects_id_range_and_limit() -> None:
    tasks = {f"HumanEval/{idx}": {"prompt": str(idx)} for idx in range(10)}

    selected = select_evalplus_tasks(tasks, limit=3, id_start=2, id_end=8)

    assert [task_id for task_id, _ in selected] == ["HumanEval/2", "HumanEval/3", "HumanEval/4"]


def test_select_evalplus_tasks_respects_explicit_task_ids() -> None:
    tasks = {f"HumanEval/{idx}": {"prompt": str(idx)} for idx in range(10)}

    selected = select_evalplus_tasks(
        tasks,
        task_ids=["HumanEval/3", "HumanEval/7"],
    )

    assert [task_id for task_id, _ in selected] == ["HumanEval/3", "HumanEval/7"]


def test_strip_markdown_code_fence_prefers_python_block() -> None:
    text = "Here is code:\n```python\ndef f():\n    return 1\n```\n"

    assert strip_markdown_code_fence(text) == "def f():\n    return 1"


def test_build_evalplus_prompt_requests_code_only() -> None:
    prompt = build_evalplus_prompt("def add(a, b):\n    pass")

    assert "Return only valid Python code" in prompt
    assert "def add" in prompt


def test_build_evalplus_prompt_literal_style_requests_exact_behavior() -> None:
    prompt = build_evalplus_prompt("def add(a, b):\n    pass", style="literal")

    assert "exact behavior" in prompt
    assert "standard library" in prompt
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


def test_select_evalplus_by_base_tests_can_tie_break_by_length(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        '{"task_id": "HumanEval/0", "solution": "short"}\n'
        '{"task_id": "HumanEval/0", "solution": "much longer solution"}\n',
        encoding="utf-8",
    )
    eval_results = tmp_path / "results.json"
    eval_results.write_text(
        """
{
  "eval": {
    "HumanEval/0": [
      {"base_status": "pass", "plus_status": "fail"},
      {"base_status": "pass", "plus_status": "pass"}
    ]
  }
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "selected.jsonl"

    report = select_evalplus_by_base_tests(
        samples,
        eval_results,
        output,
        tie_breaker="longest",
    )

    assert report.selected_base_pass == 1
    assert report.selected_plus_pass == 1
    assert output.read_text(encoding="utf-8").strip() == (
        '{"task_id": "HumanEval/0", "solution": "much longer solution"}'
    )


def test_choose_evalplus_candidate_index_rejects_unknown_tie_breaker() -> None:
    candidates = [{"solution": "a"}]

    try:
        choose_evalplus_candidate_index(candidates, [0], tie_breaker="unknown")
    except ValueError as exc:
        assert "tie_breaker" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_run_prompt_doctests() -> None:
    prompt = '''def add_one(x):
    """
    >>> add_one(1)
    2
    """
'''
    good = "def add_one(x):\n    return x + 1\n"
    bad = "def add_one(x):\n    return x\n"

    assert run_prompt_doctests(good, prompt).failures == 0
    assert run_prompt_doctests(bad, prompt).failures == 1


def test_select_evalplus_by_prompt_doctests(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "HumanEval/0", "solution": "def add_one(x):\n    return x\n"}),
                json.dumps(
                    {"task_id": "HumanEval/0", "solution": "def add_one(x):\n    return x + 1\n"}
                ),
                json.dumps(
                    {"task_id": "HumanEval/1", "solution": "def no_examples(x):\n    return x\n"}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tasks = {
        "HumanEval/0": {
            "prompt": 'def add_one(x):\n    """\n    >>> add_one(1)\n    2\n    """\n',
        },
        "HumanEval/1": {
            "prompt": "def no_examples(x):\n    pass\n",
        },
    }
    output = tmp_path / "selected.jsonl"

    report = select_evalplus_by_prompt_doctests_from_tasks(samples, tasks, output)

    assert report.selected_base_pass == 1
    assert report.fallback_tasks == ["HumanEval/1"]
    assert output.read_text(encoding="utf-8").splitlines() == [
        json.dumps({"task_id": "HumanEval/0", "solution": "def add_one(x):\n    return x + 1\n"}),
        json.dumps({"task_id": "HumanEval/1", "solution": "def no_examples(x):\n    return x\n"}),
    ]
