from __future__ import annotations

from l20_codeforge.data.code_bench_sft import build_mbpp_prompt


def test_build_mbpp_prompt_includes_problem_and_tests() -> None:
    prompt = build_mbpp_prompt(
        {
            "text": "Write a function that adds two numbers.",
            "test_list": ["assert add(1, 2) == 3"],
        }
    )

    assert "Write a function" in prompt
    assert "assert add(1, 2) == 3" in prompt
    assert "Return only valid Python code" in prompt
