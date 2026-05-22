from __future__ import annotations

import difflib
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
    test_files: dict[str, str]


SMOKE_TASKS = [
    SmokeTaskDefinition(
        task_id="py_slugify_collapse_separators",
        issue=(
            "Fix `slugify` so it lowercases text, replaces every run of non-alphanumeric "
            "characters with one hyphen, and strips leading/trailing hyphens."
        ),
        buggy_files={
            "string_utils.py": '''\
import re


def slugify(text: str) -> str:
    """Return a URL-friendly slug."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", text.lower())
''',
        },
        fixed_files={
            "string_utils.py": '''\
import re


def slugify(text: str) -> str:
    """Return a URL-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower())
    return slug.strip("-")
''',
        },
        test_files={
            "tests/test_string_utils.py": '''\
import unittest

from string_utils import slugify


class SlugifyTest(unittest.TestCase):
    def test_strips_outer_separators(self):
        self.assertEqual(slugify("  Hello, L20!! "), "hello-l20")

    def test_collapses_internal_separators(self):
        self.assertEqual(slugify("Fast---Code___RL"), "fast-code-rl")


if __name__ == "__main__":
    unittest.main()
''',
        },
    ),
    SmokeTaskDefinition(
        task_id="py_chunked_keeps_tail",
        issue=(
            "Fix `chunked` so it returns all fixed-size chunks and preserves the final "
            "short tail chunk instead of dropping it."
        ),
        buggy_files={
            "sequence_utils.py": '''\
from collections.abc import Sequence


def chunked(items: Sequence[int], size: int) -> list[list[int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    chunks = []
    for start in range(0, len(items) - size + 1, size):
        chunks.append(list(items[start : start + size]))
    return chunks
''',
        },
        fixed_files={
            "sequence_utils.py": '''\
from collections.abc import Sequence


def chunked(items: Sequence[int], size: int) -> list[list[int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    chunks = []
    for start in range(0, len(items), size):
        chunks.append(list(items[start : start + size]))
    return chunks
''',
        },
        test_files={
            "tests/test_sequence_utils.py": '''\
import unittest

from sequence_utils import chunked


class ChunkedTest(unittest.TestCase):
    def test_keeps_tail(self):
        self.assertEqual(chunked([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])

    def test_rejects_zero(self):
        with self.assertRaises(ValueError):
            chunked([1], 0)


if __name__ == "__main__":
    unittest.main()
''',
        },
    ),
    SmokeTaskDefinition(
        task_id="py_parse_ints_ignores_blanks",
        issue=(
            "Fix `parse_ints` so comma-separated blank fields are ignored while malformed "
            "non-blank values still raise ValueError."
        ),
        buggy_files={
            "parse_utils.py": '''\
def parse_ints(csv_text: str) -> list[int]:
    return [int(part) for part in csv_text.split(",")]
''',
        },
        fixed_files={
            "parse_utils.py": '''\
def parse_ints(csv_text: str) -> list[int]:
    return [int(part) for part in csv_text.split(",") if part.strip()]
''',
        },
        test_files={
            "tests/test_parse_utils.py": '''\
import unittest

from parse_utils import parse_ints


class ParseIntsTest(unittest.TestCase):
    def test_ignores_blank_fields(self):
        self.assertEqual(parse_ints("1, 2,,3, "), [1, 2, 3])

    def test_malformed_value_still_raises(self):
        with self.assertRaises(ValueError):
            parse_ints("1,nope,3")


if __name__ == "__main__":
    unittest.main()
''',
        },
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
        (repo_dir / "tests").mkdir(exist_ok=True)

        for rel_path, content in definition.buggy_files.items():
            _write_text(repo_dir / rel_path, content)
        for rel_path, content in definition.test_files.items():
            _write_text(repo_dir / rel_path, content)
        _write_text(
            repo_dir / "README.md",
            f"# {definition.task_id}\n\n{definition.issue}\n",
        )

        patch = _make_patch(definition.buggy_files, definition.fixed_files)
        patch_path = task_dir / "reference.patch"
        _write_text(patch_path, patch)

        task = TaskSpec(
            task_id=definition.task_id,
            repo=str(repo_dir),
            issue=definition.issue,
            visible_test_command="python3 -m unittest discover -s tests",
            allowed_commands=["python3 -m unittest discover -s tests"],
            metadata={"reference_patch": str(patch_path)},
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
