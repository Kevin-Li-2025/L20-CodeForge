from __future__ import annotations

from pathlib import Path

from l20_codeforge.context.compiler import ContextCompiler


def test_context_compiler_prioritizes_query_terms(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "solver.py").write_text("def solve_bug():\n    return 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("general guide\n", encoding="utf-8")

    pack = ContextCompiler(tmp_path, budget_chars=1000).compile("solve bug")

    assert pack.files
    assert pack.files[0].path == "src/solver.py"

