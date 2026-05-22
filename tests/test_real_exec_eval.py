from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from l20_codeforge.evals.real_exec import (
    evaluate_real_patch,
    infer_test_command,
    normalize_unittest_label,
    parse_swebench_tests,
)


def test_parse_swebench_tests_flattens_json_string_lists() -> None:
    assert parse_swebench_tests(['["a", "b"]']) == ["a", "b"]
    assert parse_swebench_tests(["a", ["b", '["c"]']]) == ["a", "b", "c"]
    assert parse_swebench_tests(None) == []


def test_normalize_unittest_label_handles_verbose_django_label() -> None:
    assert (
        normalize_unittest_label("test_callable_path (model_fields.test_filepathfield.FilePathFieldTests)")
        == "model_fields.test_filepathfield.FilePathFieldTests.test_callable_path"
    )


def test_infer_test_command_uses_django_runner() -> None:
    payload = {
        "dataset": "swe-bench-lite",
        "split": "test",
        "instance_id": "django__django-10924",
        "repo": "django/django",
        "base_commit": "abc",
        "FAIL_TO_PASS": (
            '["test_callable_path (model_fields.test_filepathfield.FilePathFieldTests)", '
            '"test_other (model_fields.test_filepathfield.FilePathFieldTests)"]'
        ),
    }
    from l20_codeforge.data.real_datasets import normalize_real_row

    record = normalize_real_row(
        spec=type(
            "Spec",
            (),
            {"key": "swe-bench-lite", "language": "python"},
        )(),
        split="test",
        row=payload,
    )

    assert infer_test_command(record) == (
        "PYTHONPATH=. python tests/runtests.py "
        "model_fields.test_filepathfield.FilePathFieldTests.test_callable_path "
        "model_fields.test_filepathfield.FilePathFieldTests.test_other "
        "--verbosity=2 --noinput"
    )


def test_evaluate_real_patch_runs_base_gold_and_candidate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "bug.py").write_text("def answer():\n    return 41\n", encoding="utf-8")
    _git(repo, "add", "bug.py")
    _git(repo, "commit", "-m", "base")
    base_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

    test_patch = """diff --git a/test_bug.py b/test_bug.py
new file mode 100644
--- /dev/null
+++ b/test_bug.py
@@ -0,0 +1,5 @@
+from bug import answer
+
+
+def test_answer():
+    assert answer() == 42
"""
    gold_patch = """diff --git a/bug.py b/bug.py
--- a/bug.py
+++ b/bug.py
@@ -1,2 +1,2 @@
 def answer():
-    return 41
+    return 42
"""

    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        json.dumps(
            {
                "dataset": "local",
                "split": "test",
                "instance_id": "local__repo-1",
                "repo": "local/repo",
                "base_commit": base_commit,
                "problem_statement": "Return the expected answer.",
                "patch": gold_patch,
                "test_patch": test_patch,
                "fail_to_pass": ["test_bug.py::test_answer"],
                "pass_to_pass": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_patch = tmp_path / "candidate.patch"
    candidate_patch.write_text(gold_patch, encoding="utf-8")

    report = evaluate_real_patch(
        real_tasks_path=tasks_path,
        instance_id="local__repo-1",
        patch_file=candidate_patch,
        output=tmp_path / "report.json",
        repo_dir=repo,
        test_command=f"{sys.executable} -m pytest test_bug.py::test_answer -q",
        timeout_seconds=30,
    )

    assert report.verdict == "resolved"
    assert report.base_result.status == "failed"
    assert report.gold_result.status == "passed"
    assert report.candidate_result.status == "passed"
    assert (tmp_path / "report.json").exists()
    assert _git(repo, "status", "--short").stdout == ""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
