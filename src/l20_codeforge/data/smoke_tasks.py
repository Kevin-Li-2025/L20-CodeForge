from __future__ import annotations

import difflib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from l20_codeforge.data.io import write_json
from l20_codeforge.data.schema import TaskSpec


@dataclass(frozen=True)
class SmokeTaskDefinition:
    task_id: str
    issue: str
    buggy_files: dict[str, str]
    fixed_files: dict[str, str]
    visible_test_files: dict[str, str]
    hidden_test_files: dict[str, str]
    tags: tuple[str, ...] = ()


def py_task(
    task_id: str,
    issue: str,
    module_path: str,
    buggy: str,
    fixed: str,
    visible_tests: str,
    hidden_tests: str,
    tags: tuple[str, ...] = (),
) -> SmokeTaskDefinition:
    return SmokeTaskDefinition(
        task_id=task_id,
        issue=issue,
        buggy_files={module_path: _clean(buggy)},
        fixed_files={module_path: _clean(fixed)},
        visible_test_files={f"tests_visible/test_{module_path.removesuffix('.py')}.py": _clean(visible_tests)},
        hidden_test_files={f"tests_hidden/test_{module_path.removesuffix('.py')}.py": _clean(hidden_tests)},
        tags=tags,
    )


def _clean(text: str) -> str:
    text = text.strip("\n")
    return text + "\n"


SMOKE_TASKS = [
    py_task(
        "py_slugify_strip_edges",
        "Fix `slugify` so it strips leading and trailing hyphens after replacing separators.",
        "string_utils.py",
        '''
import re


def slugify(text: str) -> str:
    """Return a URL-friendly slug."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", text.lower())
''',
        '''
import re


def slugify(text: str) -> str:
    """Return a URL-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower())
    return slug.strip("-")
''',
        '''
import unittest
from string_utils import slugify


class SlugifyVisibleTest(unittest.TestCase):
    def test_strips_outer_separators(self):
        self.assertEqual(slugify("  Hello, L20!! "), "hello-l20")


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from string_utils import slugify


class SlugifyHiddenTest(unittest.TestCase):
    def test_punctuation_only_returns_empty(self):
        self.assertEqual(slugify("!!!"), "")

    def test_keeps_numbers(self):
        self.assertEqual(slugify(" Model 7B "), "model-7b")


if __name__ == "__main__":
    unittest.main()
''',
        ("string", "normalization"),
    ),
    py_task(
        "py_chunked_keeps_tail",
        "Fix `chunked` so it preserves the final short tail chunk instead of dropping it.",
        "sequence_utils.py",
        '''
from collections.abc import Sequence


def chunked(items: Sequence[int], size: int) -> list[list[int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    chunks = []
    for start in range(0, len(items) - size + 1, size):
        chunks.append(list(items[start : start + size]))
    return chunks
''',
        '''
from collections.abc import Sequence


def chunked(items: Sequence[int], size: int) -> list[list[int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    chunks = []
    for start in range(0, len(items), size):
        chunks.append(list(items[start : start + size]))
    return chunks
''',
        '''
import unittest
from sequence_utils import chunked


class ChunkedVisibleTest(unittest.TestCase):
    def test_keeps_tail(self):
        self.assertEqual(chunked([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from sequence_utils import chunked


class ChunkedHiddenTest(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(chunked([], 3), [])

    def test_rejects_zero(self):
        with self.assertRaises(ValueError):
            chunked([1], 0)


if __name__ == "__main__":
    unittest.main()
''',
        ("sequence", "edge-case"),
    ),
    py_task(
        "py_parse_ints_ignore_blank_fields",
        "Fix `parse_ints` so comma-separated blank fields are ignored while malformed values still fail.",
        "parse_utils.py",
        '''
def parse_ints(csv_text: str) -> list[int]:
    return [int(part) for part in csv_text.split(",")]
''',
        '''
def parse_ints(csv_text: str) -> list[int]:
    return [int(part) for part in csv_text.split(",") if part.strip()]
''',
        '''
import unittest
from parse_utils import parse_ints


class ParseIntsVisibleTest(unittest.TestCase):
    def test_ignores_blank_fields(self):
        self.assertEqual(parse_ints("1, 2,,3, "), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from parse_utils import parse_ints


class ParseIntsHiddenTest(unittest.TestCase):
    def test_all_blank_is_empty(self):
        self.assertEqual(parse_ints(" ,,, "), [])

    def test_malformed_value_still_raises(self):
        with self.assertRaises(ValueError):
            parse_ints("1,nope,3")


if __name__ == "__main__":
    unittest.main()
''',
        ("parsing", "validation"),
    ),
    py_task(
        "py_median_even_average",
        "Fix `median` so even-length inputs return the average of the two middle values.",
        "stats_utils.py",
        '''
def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    return ordered[len(ordered) // 2]
''',
        '''
def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
''',
        '''
import unittest
from stats_utils import median


class MedianVisibleTest(unittest.TestCase):
    def test_even_count(self):
        self.assertEqual(median([10, 2, 4, 8]), 6)


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from stats_utils import median


class MedianHiddenTest(unittest.TestCase):
    def test_odd_count(self):
        self.assertEqual(median([3, 1, 2]), 2)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            median([])


if __name__ == "__main__":
    unittest.main()
''',
        ("math", "statistics"),
    ),
    py_task(
        "py_moving_average_includes_last_window",
        "Fix `moving_average` so it includes the final valid window.",
        "stats_utils.py",
        '''
def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    return [sum(values[i : i + window]) / window for i in range(len(values) - window)]
''',
        '''
def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    return [sum(values[i : i + window]) / window for i in range(len(values) - window + 1)]
''',
        '''
import unittest
from stats_utils import moving_average


class MovingAverageVisibleTest(unittest.TestCase):
    def test_includes_last_window(self):
        self.assertEqual(moving_average([2, 4, 6], 2), [3, 5])


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from stats_utils import moving_average


class MovingAverageHiddenTest(unittest.TestCase):
    def test_exact_window(self):
        self.assertEqual(moving_average([1, 2, 3], 3), [2])

    def test_bad_window(self):
        with self.assertRaises(ValueError):
            moving_average([1], 0)


if __name__ == "__main__":
    unittest.main()
''',
        ("math", "off-by-one"),
    ),
    py_task(
        "py_dedupe_preserve_order",
        "Fix `dedupe` so it removes duplicates while preserving first-seen order.",
        "sequence_utils.py",
        '''
def dedupe(items: list[str]) -> list[str]:
    return list(set(items))
''',
        '''
def dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
''',
        '''
import unittest
from sequence_utils import dedupe


class DedupeVisibleTest(unittest.TestCase):
    def test_preserves_order(self):
        self.assertEqual(dedupe(["b", "a", "b", "c", "a"]), ["b", "a", "c"])


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from sequence_utils import dedupe


class DedupeHiddenTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(dedupe([]), [])

    def test_already_unique(self):
        self.assertEqual(dedupe(["x", "y"]), ["x", "y"])


if __name__ == "__main__":
    unittest.main()
''',
        ("sequence", "order"),
    ),
    py_task(
        "py_flatten_one_level_skips_strings",
        "Fix `flatten_one_level` so strings are treated as scalar values, not iterables to expand.",
        "sequence_utils.py",
        '''
from collections.abc import Iterable


def flatten_one_level(items):
    result = []
    for item in items:
        if isinstance(item, Iterable):
            result.extend(item)
        else:
            result.append(item)
    return result
''',
        '''
from collections.abc import Iterable


def flatten_one_level(items):
    result = []
    for item in items:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            result.extend(item)
        else:
            result.append(item)
    return result
''',
        '''
import unittest
from sequence_utils import flatten_one_level


class FlattenVisibleTest(unittest.TestCase):
    def test_keeps_string_scalar(self):
        self.assertEqual(flatten_one_level([[1, 2], "ab", [3]]), [1, 2, "ab", 3])


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from sequence_utils import flatten_one_level


class FlattenHiddenTest(unittest.TestCase):
    def test_bytes_scalar(self):
        self.assertEqual(flatten_one_level([b"xy", [1]]), [b"xy", 1])


if __name__ == "__main__":
    unittest.main()
''',
        ("sequence", "typing"),
    ),
    py_task(
        "py_deep_get_allows_falsy_values",
        "Fix `deep_get` so falsy existing values like 0 and False are returned instead of default.",
        "dict_utils.py",
        '''
def deep_get(data: dict, path: list[str], default=None):
    current = data
    for key in path:
        current = current.get(key)
        if not current:
            return default
    return current
''',
        '''
def deep_get(data: dict, path: list[str], default=None):
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
''',
        '''
import unittest
from dict_utils import deep_get


class DeepGetVisibleTest(unittest.TestCase):
    def test_returns_zero(self):
        self.assertEqual(deep_get({"a": {"b": 0}}, ["a", "b"], "missing"), 0)


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from dict_utils import deep_get


class DeepGetHiddenTest(unittest.TestCase):
    def test_returns_false(self):
        self.assertIs(deep_get({"a": {"b": False}}, ["a", "b"], "missing"), False)

    def test_missing_default(self):
        self.assertEqual(deep_get({"a": {}}, ["a", "x"], "missing"), "missing")


if __name__ == "__main__":
    unittest.main()
''',
        ("dict", "falsy"),
    ),
    py_task(
        "py_invert_multimap_keeps_all_keys",
        "Fix `invert_multimap` so duplicate values map to all source keys.",
        "dict_utils.py",
        '''
def invert_multimap(data: dict[str, str]) -> dict[str, list[str]]:
    result = {}
    for key, value in data.items():
        result[value] = [key]
    return result
''',
        '''
def invert_multimap(data: dict[str, str]) -> dict[str, list[str]]:
    result = {}
    for key, value in data.items():
        result.setdefault(value, []).append(key)
    return result
''',
        '''
import unittest
from dict_utils import invert_multimap


class InvertVisibleTest(unittest.TestCase):
    def test_keeps_all_keys_for_value(self):
        self.assertEqual(invert_multimap({"a": "x", "b": "x"}), {"x": ["a", "b"]})


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from dict_utils import invert_multimap


class InvertHiddenTest(unittest.TestCase):
    def test_multiple_values(self):
        self.assertEqual(invert_multimap({"a": "x", "b": "y", "c": "x"}), {"x": ["a", "c"], "y": ["b"]})


if __name__ == "__main__":
    unittest.main()
''',
        ("dict", "aggregation"),
    ),
    py_task(
        "py_merge_counts_adds_values",
        "Fix `merge_counts` so counts from both dictionaries are added instead of overwritten.",
        "dict_utils.py",
        '''
def merge_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = dict(left)
    merged.update(right)
    return merged
''',
        '''
def merge_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = merged.get(key, 0) + value
    return merged
''',
        '''
import unittest
from dict_utils import merge_counts


class MergeCountsVisibleTest(unittest.TestCase):
    def test_adds_overlapping_counts(self):
        self.assertEqual(merge_counts({"a": 2}, {"a": 3}), {"a": 5})


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from dict_utils import merge_counts


class MergeCountsHiddenTest(unittest.TestCase):
    def test_keeps_non_overlapping(self):
        self.assertEqual(merge_counts({"a": 1}, {"b": 4}), {"a": 1, "b": 4})


if __name__ == "__main__":
    unittest.main()
''',
        ("dict", "aggregation"),
    ),
    py_task(
        "py_normalize_whitespace_strips_edges",
        "Fix `normalize_whitespace` so it collapses whitespace and strips leading/trailing spaces.",
        "string_utils.py",
        '''
import re


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\\s+", " ", text)
''',
        '''
import re


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\\s+", " ", text).strip()
''',
        '''
import unittest
from string_utils import normalize_whitespace


class NormalizeVisibleTest(unittest.TestCase):
    def test_strips_edges(self):
        self.assertEqual(normalize_whitespace("  a\\n b\\t "), "a b")


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from string_utils import normalize_whitespace


class NormalizeHiddenTest(unittest.TestCase):
    def test_empty_after_spaces(self):
        self.assertEqual(normalize_whitespace(" \\t\\n "), "")


if __name__ == "__main__":
    unittest.main()
''',
        ("string", "normalization"),
    ),
    py_task(
        "py_truncate_adds_ellipsis_within_limit",
        "Fix `truncate` so the ellipsis is included within the maximum length.",
        "string_utils.py",
        '''
def truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
''',
        '''
def truncate(text: str, max_length: int) -> str:
    if max_length < 0:
        raise ValueError("max_length must be non-negative")
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return "." * max_length
    return text[: max_length - 3] + "..."
''',
        '''
import unittest
from string_utils import truncate


class TruncateVisibleTest(unittest.TestCase):
    def test_ellipsis_within_limit(self):
        self.assertEqual(truncate("abcdef", 5), "ab...")


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from string_utils import truncate


class TruncateHiddenTest(unittest.TestCase):
    def test_short_limit(self):
        self.assertEqual(truncate("abcdef", 2), "..")

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            truncate("abc", -1)


if __name__ == "__main__":
    unittest.main()
''',
        ("string", "boundary"),
    ),
    py_task(
        "py_safe_divide_allows_zero_numerator",
        "Fix `safe_divide` so a zero numerator returns 0 instead of default.",
        "math_utils.py",
        '''
def safe_divide(numerator: float, denominator: float, default=None):
    if not numerator or not denominator:
        return default
    return numerator / denominator
''',
        '''
def safe_divide(numerator: float, denominator: float, default=None):
    if denominator == 0:
        return default
    return numerator / denominator
''',
        '''
import unittest
from math_utils import safe_divide


class SafeDivideVisibleTest(unittest.TestCase):
    def test_zero_numerator_is_zero(self):
        self.assertEqual(safe_divide(0, 5, "bad"), 0)


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from math_utils import safe_divide


class SafeDivideHiddenTest(unittest.TestCase):
    def test_zero_denominator_default(self):
        self.assertEqual(safe_divide(5, 0, "bad"), "bad")


if __name__ == "__main__":
    unittest.main()
''',
        ("math", "falsy"),
    ),
    py_task(
        "py_clamp_swaps_reversed_bounds",
        "Fix `clamp` so reversed min/max bounds are handled by swapping them.",
        "math_utils.py",
        '''
def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)
''',
        '''
def clamp(value: float, minimum: float, maximum: float) -> float:
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    return min(max(value, minimum), maximum)
''',
        '''
import unittest
from math_utils import clamp


class ClampVisibleTest(unittest.TestCase):
    def test_reversed_bounds(self):
        self.assertEqual(clamp(5, 10, 0), 5)


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from math_utils import clamp


class ClampHiddenTest(unittest.TestCase):
    def test_clamps_low_after_swap(self):
        self.assertEqual(clamp(-2, 10, 0), 0)


if __name__ == "__main__":
    unittest.main()
''',
        ("math", "boundary"),
    ),
    py_task(
        "py_parse_bool_accepts_common_values",
        "Fix `parse_bool` so it accepts common yes/no strings case-insensitively.",
        "parse_utils.py",
        '''
def parse_bool(text: str) -> bool:
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"invalid bool: {text}")
''',
        '''
def parse_bool(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid bool: {text}")
''',
        '''
import unittest
from parse_utils import parse_bool


class ParseBoolVisibleTest(unittest.TestCase):
    def test_yes_no_values(self):
        self.assertTrue(parse_bool(" YES "))
        self.assertFalse(parse_bool("no"))


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from parse_utils import parse_bool


class ParseBoolHiddenTest(unittest.TestCase):
    def test_numeric_values(self):
        self.assertTrue(parse_bool("1"))
        self.assertFalse(parse_bool("0"))

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            parse_bool("maybe")


if __name__ == "__main__":
    unittest.main()
''',
        ("parsing", "normalization"),
    ),
    py_task(
        "py_parse_kv_trims_parts",
        "Fix `parse_kv` so keys and values are stripped and values may contain '='.",
        "parse_utils.py",
        '''
def parse_kv(line: str) -> tuple[str, str]:
    key, value = line.split("=")
    return key, value
''',
        '''
def parse_kv(line: str) -> tuple[str, str]:
    key, value = line.split("=", 1)
    return key.strip(), value.strip()
''',
        '''
import unittest
from parse_utils import parse_kv


class ParseKvVisibleTest(unittest.TestCase):
    def test_trims_parts(self):
        self.assertEqual(parse_kv(" name = L20 "), ("name", "L20"))


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from parse_utils import parse_kv


class ParseKvHiddenTest(unittest.TestCase):
    def test_value_can_contain_equals(self):
        self.assertEqual(parse_kv("expr=a=b"), ("expr", "a=b"))


if __name__ == "__main__":
    unittest.main()
''',
        ("parsing", "delimiter"),
    ),
    py_task(
        "py_top_k_sorts_by_score_desc",
        "Fix `top_k` so it returns the k highest-scoring pairs in descending score order.",
        "ranking_utils.py",
        '''
def top_k(items: list[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    return sorted(items, key=lambda item: item[1])[:k]
''',
        '''
def top_k(items: list[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    if k <= 0:
        return []
    return sorted(items, key=lambda item: item[1], reverse=True)[:k]
''',
        '''
import unittest
from ranking_utils import top_k


class TopKVisibleTest(unittest.TestCase):
    def test_highest_first(self):
        self.assertEqual(top_k([("a", 1), ("b", 3), ("c", 2)], 2), [("b", 3), ("c", 2)])


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from ranking_utils import top_k


class TopKHiddenTest(unittest.TestCase):
    def test_zero_k(self):
        self.assertEqual(top_k([("a", 1)], 0), [])


if __name__ == "__main__":
    unittest.main()
''',
        ("ranking", "sorting"),
    ),
    py_task(
        "py_argmax_returns_first_tie",
        "Fix `argmax` so ties return the first matching key, not the last.",
        "ranking_utils.py",
        '''
def argmax(scores: dict[str, float]) -> str:
    if not scores:
        raise ValueError("scores cannot be empty")
    best_key = None
    best_score = float("-inf")
    for key, score in scores.items():
        if score >= best_score:
            best_key = key
            best_score = score
    return best_key
''',
        '''
def argmax(scores: dict[str, float]) -> str:
    if not scores:
        raise ValueError("scores cannot be empty")
    best_key = None
    best_score = float("-inf")
    for key, score in scores.items():
        if score > best_score:
            best_key = key
            best_score = score
    return best_key
''',
        '''
import unittest
from ranking_utils import argmax


class ArgmaxVisibleTest(unittest.TestCase):
    def test_returns_first_tie(self):
        self.assertEqual(argmax({"a": 2, "b": 2}), "a")


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from ranking_utils import argmax


class ArgmaxHiddenTest(unittest.TestCase):
    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            argmax({})


if __name__ == "__main__":
    unittest.main()
''',
        ("ranking", "tie-break"),
    ),
    py_task(
        "py_token_f1_handles_empty_prediction",
        "Fix `token_f1` so empty prediction and empty truth both score 1.0.",
        "metrics.py",
        '''
def token_f1(prediction: list[str], truth: list[str]) -> float:
    overlap = len(set(prediction) & set(truth))
    precision = overlap / len(prediction)
    recall = overlap / len(truth)
    return 2 * precision * recall / (precision + recall)
''',
        '''
def token_f1(prediction: list[str], truth: list[str]) -> float:
    if not prediction and not truth:
        return 1.0
    if not prediction or not truth:
        return 0.0
    overlap = len(set(prediction) & set(truth))
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction)
    recall = overlap / len(truth)
    return 2 * precision * recall / (precision + recall)
''',
        '''
import unittest
from metrics import token_f1


class TokenF1VisibleTest(unittest.TestCase):
    def test_both_empty(self):
        self.assertEqual(token_f1([], []), 1.0)


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from metrics import token_f1


class TokenF1HiddenTest(unittest.TestCase):
    def test_one_empty(self):
        self.assertEqual(token_f1(["a"], []), 0.0)

    def test_no_overlap(self):
        self.assertEqual(token_f1(["a"], ["b"]), 0.0)


if __name__ == "__main__":
    unittest.main()
''',
        ("metrics", "zero-division"),
    ),
    py_task(
        "py_accuracy_uses_total_examples",
        "Fix `accuracy` so it divides by the total number of labels, not the number of correct predictions.",
        "metrics.py",
        '''
def accuracy(predictions: list[str], labels: list[str]) -> float:
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return correct / len(predictions)
''',
        '''
def accuracy(predictions: list[str], labels: list[str]) -> float:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not labels:
        raise ValueError("labels cannot be empty")
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return correct / len(labels)
''',
        '''
import unittest
from metrics import accuracy


class AccuracyVisibleTest(unittest.TestCase):
    def test_basic_accuracy(self):
        self.assertEqual(accuracy(["a", "b", "x"], ["a", "b", "c"]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from metrics import accuracy


class AccuracyHiddenTest(unittest.TestCase):
    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            accuracy(["a"], ["a", "b"])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            accuracy([], [])


if __name__ == "__main__":
    unittest.main()
''',
        ("metrics", "validation"),
    ),
    py_task(
        "py_retry_stops_after_success",
        "Fix `retry` so it stops immediately after the callable succeeds.",
        "control_utils.py",
        '''
def retry(fn, attempts: int):
    result = None
    for _ in range(attempts):
        result = fn()
    return result
''',
        '''
def retry(fn, attempts: int):
    last_error = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("attempts must be positive")
''',
        '''
import unittest
from control_utils import retry


class RetryVisibleTest(unittest.TestCase):
    def test_stops_after_success(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        self.assertEqual(retry(fn, 3), "ok")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from control_utils import retry


class RetryHiddenTest(unittest.TestCase):
    def test_retries_after_failure(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("try again")
            return "ok"

        self.assertEqual(retry(fn, 2), "ok")


if __name__ == "__main__":
    unittest.main()
''',
        ("control-flow", "exceptions"),
    ),
    py_task(
        "py_window_pairs_includes_adjacent_pairs",
        "Fix `window_pairs` so it returns every adjacent pair.",
        "sequence_utils.py",
        '''
def window_pairs(items: list[int]) -> list[tuple[int, int]]:
    return [(items[i], items[i + 1]) for i in range(len(items) - 2)]
''',
        '''
def window_pairs(items: list[int]) -> list[tuple[int, int]]:
    return [(items[i], items[i + 1]) for i in range(len(items) - 1)]
''',
        '''
import unittest
from sequence_utils import window_pairs


class WindowPairsVisibleTest(unittest.TestCase):
    def test_all_adjacent_pairs(self):
        self.assertEqual(window_pairs([1, 2, 3]), [(1, 2), (2, 3)])


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from sequence_utils import window_pairs


class WindowPairsHiddenTest(unittest.TestCase):
    def test_single_item(self):
        self.assertEqual(window_pairs([1]), [])


if __name__ == "__main__":
    unittest.main()
''',
        ("sequence", "off-by-one"),
    ),
    py_task(
        "py_group_by_accumulates_all_items",
        "Fix `group_by` so each key accumulates all matching items.",
        "group_utils.py",
        '''
def group_by(items, key_fn):
    groups = {}
    for item in items:
        groups[key_fn(item)] = [item]
    return groups
''',
        '''
def group_by(items, key_fn):
    groups = {}
    for item in items:
        groups.setdefault(key_fn(item), []).append(item)
    return groups
''',
        '''
import unittest
from group_utils import group_by


class GroupByVisibleTest(unittest.TestCase):
    def test_accumulates(self):
        self.assertEqual(group_by(["a", "bb", "c"], len), {1: ["a", "c"], 2: ["bb"]})


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from group_utils import group_by


class GroupByHiddenTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(group_by([], len), {})


if __name__ == "__main__":
    unittest.main()
''',
        ("aggregation", "dict"),
    ),
    py_task(
        "py_format_table_uses_all_rows",
        "Fix `format_table` so it includes every row, not just rows after the first.",
        "format_utils.py",
        '''
def format_table(rows: list[list[str]]) -> str:
    return "\\n".join(" | ".join(row) for row in rows[1:])
''',
        '''
def format_table(rows: list[list[str]]) -> str:
    return "\\n".join(" | ".join(row) for row in rows)
''',
        '''
import unittest
from format_utils import format_table


class FormatTableVisibleTest(unittest.TestCase):
    def test_includes_header_row(self):
        self.assertEqual(format_table([["a", "b"], ["1", "2"]]), "a | b\\n1 | 2")


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from format_utils import format_table


class FormatTableHiddenTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(format_table([]), "")


if __name__ == "__main__":
    unittest.main()
''',
        ("formatting", "off-by-one"),
    ),
    py_task(
        "py_parse_lines_skips_comments",
        "Fix `parse_lines` so blank lines and comment lines are skipped.",
        "parse_utils.py",
        '''
def parse_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]
''',
        '''
def parse_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
''',
        '''
import unittest
from parse_utils import parse_lines


class ParseLinesVisibleTest(unittest.TestCase):
    def test_skips_comments(self):
        self.assertEqual(parse_lines("a\\n# ignore\\n b "), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from parse_utils import parse_lines


class ParseLinesHiddenTest(unittest.TestCase):
    def test_indented_comment(self):
        self.assertEqual(parse_lines("  # comment\\nvalue"), ["value"])


if __name__ == "__main__":
    unittest.main()
''',
        ("parsing", "comments"),
    ),
    py_task(
        "py_unique_sorted_handles_key",
        "Fix `unique_sorted` so it sorts unique values by the provided key.",
        "sort_utils.py",
        '''
def unique_sorted(values: list[str], key=None) -> list[str]:
    return sorted(set(values))
''',
        '''
def unique_sorted(values: list[str], key=None) -> list[str]:
    return sorted(set(values), key=key)
''',
        '''
import unittest
from sort_utils import unique_sorted


class UniqueSortedVisibleTest(unittest.TestCase):
    def test_sorts_by_key(self):
        self.assertEqual(unique_sorted(["bbb", "a", "cc"], key=len), ["a", "cc", "bbb"])


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from sort_utils import unique_sorted


class UniqueSortedHiddenTest(unittest.TestCase):
    def test_removes_duplicates(self):
        self.assertEqual(unique_sorted(["b", "a", "b"]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
''',
        ("sorting", "api"),
    ),
    py_task(
        "py_transpose_handles_ragged_rows",
        "Fix `transpose` so ragged rows are padded with None.",
        "matrix_utils.py",
        '''
def transpose(rows: list[list[int]]) -> list[list[int]]:
    return [list(col) for col in zip(*rows)]
''',
        '''
from itertools import zip_longest


def transpose(rows: list[list[int]]) -> list[list[int | None]]:
    return [list(col) for col in zip_longest(*rows, fillvalue=None)]
''',
        '''
import unittest
from matrix_utils import transpose


class TransposeVisibleTest(unittest.TestCase):
    def test_ragged_rows(self):
        self.assertEqual(transpose([[1, 2], [3]]), [[1, 3], [2, None]])


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from matrix_utils import transpose


class TransposeHiddenTest(unittest.TestCase):
    def test_rectangular(self):
        self.assertEqual(transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]])


if __name__ == "__main__":
    unittest.main()
''',
        ("matrix", "ragged"),
    ),
    py_task(
        "py_dot_product_rejects_length_mismatch",
        "Fix `dot` so vector length mismatches raise ValueError instead of silently truncating.",
        "matrix_utils.py",
        '''
def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
''',
        '''
def dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    return sum(a * b for a, b in zip(left, right))
''',
        '''
import unittest
from matrix_utils import dot


class DotVisibleTest(unittest.TestCase):
    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            dot([1, 2], [3])


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from matrix_utils import dot


class DotHiddenTest(unittest.TestCase):
    def test_dot_value(self):
        self.assertEqual(dot([1, 2], [3, 4]), 11)


if __name__ == "__main__":
    unittest.main()
''',
        ("matrix", "validation"),
    ),
    py_task(
        "py_bfs_shortest_path_marks_start_seen",
        "Fix `shortest_path` so cyclic graphs do not repeatedly enqueue the start node.",
        "graph_utils.py",
        '''
from collections import deque


def shortest_path(graph: dict[str, list[str]], start: str, goal: str) -> list[str] | None:
    queue = deque([(start, [start])])
    seen = set()
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None
''',
        '''
from collections import deque


def shortest_path(graph: dict[str, list[str]], start: str, goal: str) -> list[str] | None:
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None
''',
        '''
import unittest
from graph_utils import shortest_path


class ShortestPathVisibleTest(unittest.TestCase):
    def test_cycle_does_not_repeat_start(self):
        graph = {"a": ["b"], "b": ["a", "c"], "c": []}
        self.assertEqual(shortest_path(graph, "a", "c"), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from graph_utils import shortest_path


class ShortestPathHiddenTest(unittest.TestCase):
    def test_missing_goal(self):
        self.assertIsNone(shortest_path({"a": ["b"]}, "a", "z"))


if __name__ == "__main__":
    unittest.main()
''',
        ("graph", "cycle"),
    ),
    py_task(
        "py_topological_sort_detects_cycles",
        "Fix `topological_sort` so it raises ValueError when a cycle remains.",
        "graph_utils.py",
        '''
from collections import deque


def topological_sort(edges: dict[str, list[str]]) -> list[str]:
    indegree = {node: 0 for node in edges}
    for targets in edges.values():
        for target in targets:
            indegree[target] = indegree.get(target, 0) + 1
    queue = deque([node for node, degree in indegree.items() if degree == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for target in edges.get(node, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return order
''',
        '''
from collections import deque


def topological_sort(edges: dict[str, list[str]]) -> list[str]:
    indegree = {node: 0 for node in edges}
    for targets in edges.values():
        for target in targets:
            indegree[target] = indegree.get(target, 0) + 1
    queue = deque([node for node, degree in indegree.items() if degree == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for target in edges.get(node, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != len(indegree):
        raise ValueError("graph contains a cycle")
    return order
''',
        '''
import unittest
from graph_utils import topological_sort


class TopologicalSortVisibleTest(unittest.TestCase):
    def test_detects_cycle(self):
        with self.assertRaises(ValueError):
            topological_sort({"a": ["b"], "b": ["a"]})


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from graph_utils import topological_sort


class TopologicalSortHiddenTest(unittest.TestCase):
    def test_orders_dag(self):
        order = topological_sort({"a": ["b"], "b": []})
        self.assertEqual(order, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
''',
        ("graph", "cycle"),
    ),
    py_task(
        "py_iso_date_accepts_z_suffix",
        "Fix `parse_iso_date` so a trailing Z is accepted as UTC.",
        "time_utils.py",
        '''
from datetime import datetime


def parse_iso_date(text: str) -> datetime:
    return datetime.fromisoformat(text)
''',
        '''
from datetime import datetime


def parse_iso_date(text: str) -> datetime:
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)
''',
        '''
import unittest
from time_utils import parse_iso_date


class ParseIsoDateVisibleTest(unittest.TestCase):
    def test_z_suffix(self):
        self.assertEqual(parse_iso_date("2026-05-22T00:00:00Z").utcoffset().total_seconds(), 0)


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from time_utils import parse_iso_date


class ParseIsoDateHiddenTest(unittest.TestCase):
    def test_plain_date_time(self):
        self.assertEqual(parse_iso_date("2026-05-22T01:02:03").year, 2026)


if __name__ == "__main__":
    unittest.main()
''',
        ("time", "parsing"),
    ),
    py_task(
        "py_seconds_between_abs_delta",
        "Fix `seconds_between` so it returns an absolute positive duration.",
        "time_utils.py",
        '''
from datetime import datetime


def seconds_between(start: datetime, end: datetime) -> int:
    return int((end - start).total_seconds())
''',
        '''
from datetime import datetime


def seconds_between(start: datetime, end: datetime) -> int:
    return int(abs((end - start).total_seconds()))
''',
        '''
import unittest
from datetime import datetime
from time_utils import seconds_between


class SecondsBetweenVisibleTest(unittest.TestCase):
    def test_absolute_delta(self):
        self.assertEqual(seconds_between(datetime(2026, 1, 1, 0, 0, 5), datetime(2026, 1, 1, 0, 0, 1)), 4)


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from datetime import datetime
from time_utils import seconds_between


class SecondsBetweenHiddenTest(unittest.TestCase):
    def test_forward_delta(self):
        self.assertEqual(seconds_between(datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 1)), 60)


if __name__ == "__main__":
    unittest.main()
''',
        ("time", "math"),
    ),
    py_task(
        "py_redact_email_keeps_domain",
        "Fix `redact_email` so it keeps the domain and redacts only the local part.",
        "privacy_utils.py",
        '''
def redact_email(email: str) -> str:
    return "***"
''',
        '''
def redact_email(email: str) -> str:
    local, sep, domain = email.partition("@")
    if not sep:
        return "***"
    return "***@" + domain
''',
        '''
import unittest
from privacy_utils import redact_email


class RedactEmailVisibleTest(unittest.TestCase):
    def test_keeps_domain(self):
        self.assertEqual(redact_email("user@example.com"), "***@example.com")


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from privacy_utils import redact_email


class RedactEmailHiddenTest(unittest.TestCase):
    def test_invalid_email(self):
        self.assertEqual(redact_email("not-email"), "***")


if __name__ == "__main__":
    unittest.main()
''',
        ("privacy", "string"),
    ),
    py_task(
        "py_mask_token_preserves_suffix",
        "Fix `mask_token` so it preserves the last four characters for long tokens.",
        "privacy_utils.py",
        '''
def mask_token(token: str) -> str:
    return "*" * len(token)
''',
        '''
def mask_token(token: str) -> str:
    if len(token) <= 4:
        return "*" * len(token)
    return "*" * (len(token) - 4) + token[-4:]
''',
        '''
import unittest
from privacy_utils import mask_token


class MaskTokenVisibleTest(unittest.TestCase):
    def test_preserves_suffix(self):
        self.assertEqual(mask_token("abcdef1234"), "******1234")


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from privacy_utils import mask_token


class MaskTokenHiddenTest(unittest.TestCase):
    def test_short_token_all_masked(self):
        self.assertEqual(mask_token("abc"), "***")


if __name__ == "__main__":
    unittest.main()
''',
        ("privacy", "boundary"),
    ),
    py_task(
        "py_build_query_urlencodes_values",
        "Fix `build_query` so keys and values are URL-encoded.",
        "url_utils.py",
        '''
def build_query(params: dict[str, str]) -> str:
    return "&".join(f"{key}={value}" for key, value in params.items())
''',
        '''
from urllib.parse import urlencode


def build_query(params: dict[str, str]) -> str:
    return urlencode(params)
''',
        '''
import unittest
from url_utils import build_query


class BuildQueryVisibleTest(unittest.TestCase):
    def test_encodes_space(self):
        self.assertEqual(build_query({"q": "code rl"}), "q=code+rl")


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from url_utils import build_query


class BuildQueryHiddenTest(unittest.TestCase):
    def test_encodes_ampersand(self):
        self.assertEqual(build_query({"q": "a&b"}), "q=a%26b")


if __name__ == "__main__":
    unittest.main()
''',
        ("url", "encoding"),
    ),
    py_task(
        "py_join_url_avoids_double_slash",
        "Fix `join_url` so it joins base and path with exactly one slash.",
        "url_utils.py",
        '''
def join_url(base: str, path: str) -> str:
    return base + "/" + path
''',
        '''
def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")
''',
        '''
import unittest
from url_utils import join_url


class JoinUrlVisibleTest(unittest.TestCase):
    def test_avoids_double_slash(self):
        self.assertEqual(join_url("https://x.test/", "/api"), "https://x.test/api")


if __name__ == "__main__":
    unittest.main()
''',
        '''
import unittest
from url_utils import join_url


class JoinUrlHiddenTest(unittest.TestCase):
    def test_adds_missing_slash(self):
        self.assertEqual(join_url("https://x.test", "api"), "https://x.test/api")


if __name__ == "__main__":
    unittest.main()
''',
        ("url", "normalization"),
    ),
]


def write_smoke_tasks(output_dir: Path, overwrite: bool = False) -> list[Path]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_files = []
    for definition in SMOKE_TASKS:
        task_dir = output_dir / definition.task_id
        repo_dir = task_dir / "repo"
        if task_dir.exists() and overwrite:
            shutil.rmtree(task_dir)
        repo_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / "tests_visible").mkdir(exist_ok=True)
        (repo_dir / "tests_hidden").mkdir(exist_ok=True)

        for rel_path, content in definition.buggy_files.items():
            _write_text(repo_dir / rel_path, content)
        for rel_path, content in definition.visible_test_files.items():
            _write_text(repo_dir / rel_path, content)
        for rel_path, content in definition.hidden_test_files.items():
            _write_text(repo_dir / rel_path, content)
        _write_patch_helper(repo_dir, definition.buggy_files)
        _write_text(
            repo_dir / "README.md",
            "\n".join(
                [
                    f"# {definition.task_id}",
                    "",
                    definition.issue,
                    "",
                    "Visible tests: `python3 -m unittest discover -s tests_visible`",
                    "Patch helper: `python3 .l20_codeforge/make_patch.py`",
                    f"Tags: {', '.join(definition.tags)}",
                    "",
                ]
            ),
        )

        patch = _make_patch(definition.buggy_files, definition.fixed_files)
        patch_path = task_dir / "reference.patch"
        _write_text(patch_path, patch)

        task = TaskSpec(
            task_id=definition.task_id,
            repo=str(repo_dir),
            issue=definition.issue,
            visible_test_command="python3 -m unittest discover -s tests_visible",
            hidden_test_command="python3 -m unittest discover -s tests_hidden",
            allowed_commands=["python3 -m unittest discover -s tests_visible"],
            metadata={"reference_patch": str(patch_path), "tags": ",".join(definition.tags)},
        )
        task_path = task_dir / "task.json"
        write_json(task_path, task)
        task_files.append(task_path)

    return task_files


def _make_patch(buggy_files: dict[str, str], fixed_files: dict[str, str]) -> str:
    patch_parts = []
    for rel_path, fixed in fixed_files.items():
        buggy = buggy_files[rel_path]
        patch_parts.extend(
            difflib.unified_diff(
                buggy.splitlines(keepends=True),
                fixed.splitlines(keepends=True),
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
        )
    return "".join(patch_parts)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_patch_helper(repo_dir: Path, original_files: dict[str, str]) -> None:
    helper_dir = repo_dir / ".l20_codeforge"
    source_files = sorted(original_files)
    for rel_path, content in original_files.items():
        _write_text(helper_dir / "original" / rel_path, content)
    _write_text(helper_dir / "source_files.json", json.dumps(source_files, indent=2) + "\n")
    _write_text(
        helper_dir / "make_patch.py",
        '''\
from __future__ import annotations

import difflib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / ".l20_codeforge"
SOURCE_FILES = json.loads((HELPER / "source_files.json").read_text(encoding="utf-8"))


def main() -> int:
    parts: list[str] = []
    for rel_path in SOURCE_FILES:
        original_path = HELPER / "original" / rel_path
        current_path = ROOT / rel_path
        original = original_path.read_text(encoding="utf-8").splitlines(keepends=True)
        current = current_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if original == current:
            continue
        parts.extend(
            difflib.unified_diff(
                original,
                current,
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
        )
    print("".join(parts), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
    )
