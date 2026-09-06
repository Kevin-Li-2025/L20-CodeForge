from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from l20_codeforge.artifacts import artifact_report, verify_artifacts


def test_artifact_report_distinguishes_ok_missing_and_mismatch(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_bytes(b"stable artifact\n")
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"changed artifact\n")
    expected = {
        "good.json": hashlib.sha256(good.read_bytes()).hexdigest(),
        "bad.json": hashlib.sha256(b"original artifact\n").hexdigest(),
        "missing.json": hashlib.sha256(b"missing\n").hexdigest(),
    }

    report = artifact_report(tmp_path, expected)

    assert report["status"] == "FAIL"
    statuses = {item["path"]: item["status"] for item in report["artifacts"]}
    assert statuses == {
        "good.json": "ok",
        "bad.json": "mismatch",
        "missing.json": "missing",
    }


def test_artifact_paths_cannot_escape_repository_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes root"):
        verify_artifacts(tmp_path, {"../outside.json": "0" * 64})
