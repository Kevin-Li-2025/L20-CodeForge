from __future__ import annotations

import json
from typing import Any

from l20_codeforge.rewards.code_execution import (
    CodeExecutionConfig,
    CodeTestCase,
    evaluate_python_completion,
)
from l20_codeforge.rewards.patch_reward import anti_hack_score, patch_quality_score


def patch_quality_reward(completions: list[Any], **_: Any) -> list[float]:
    return [patch_quality_score(_completion_text(item))[0] for item in completions]


def anti_hack_reward(completions: list[Any], **_: Any) -> list[float]:
    return [anti_hack_score(_completion_text(item))[0] for item in completions]


def code_execution_reward(
    completions: list[Any],
    tests: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Return dense compile/test rewards for standalone algorithmic solutions.

    The training dataset must expose a ``tests`` column. TRL normally repeats
    dataset columns to match sampled completions; a single test suite is also
    accepted for direct smoke tests.
    """

    raw_tests = tests if tests is not None else kwargs.get("test_cases")
    test_batches = _align_test_batches(raw_tests, len(completions))
    config = CodeExecutionConfig.model_validate(kwargs.get("execution_config", {}))
    return [
        evaluate_python_completion(
            _completion_text(completion),
            _parse_test_cases(test_batch),
            config,
        ).reward
        for completion, test_batch in zip(completions, test_batches, strict=True)
    ]


def code_binary_execution_reward(
    completions: list[Any],
    tests: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Return a strict all-tests-pass reward for verifier ablations."""

    raw_tests = tests if tests is not None else kwargs.get("test_cases")
    test_batches = _align_test_batches(raw_tests, len(completions))
    config = CodeExecutionConfig.model_validate(kwargs.get("execution_config", {}))
    return [
        float(
            evaluate_python_completion(
                _completion_text(completion),
                _parse_test_cases(test_batch),
                config,
            ).all_passed
        )
        for completion, test_batch in zip(completions, test_batches, strict=True)
    ]


def _completion_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, list) and item and isinstance(item[-1], dict):
        return str(item[-1].get("content", ""))
    if isinstance(item, dict):
        return str(item.get("content", item))
    return str(item)


def _align_test_batches(raw_tests: Any, completion_count: int) -> list[Any]:
    if raw_tests is None:
        raise ValueError("code execution reward requires a tests dataset column")
    if not isinstance(raw_tests, list):
        raw_tests = [raw_tests]
    if len(raw_tests) == completion_count:
        return raw_tests
    if len(raw_tests) == 1:
        return raw_tests * completion_count
    raise ValueError(
        "tests column length must equal completions length or contain one shared suite; "
        f"got {len(raw_tests)} tests batches for {completion_count} completions"
    )


def _parse_test_cases(raw_tests: Any) -> list[CodeTestCase]:
    if isinstance(raw_tests, str):
        raw_tests = json.loads(raw_tests)
    if isinstance(raw_tests, dict) and "tests" in raw_tests:
        raw_tests = raw_tests["tests"]
    if not isinstance(raw_tests, list):
        raise TypeError("each tests value must be a JSON list or an object containing tests")
    return [CodeTestCase.model_validate(item) for item in raw_tests]
