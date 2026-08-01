from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_ARTIFACT_HASHES: dict[str, str] = {
    "benchmarks/generalization_scorecard_2026_05_23/scorecard.json": (
        "1eb0402378ea25732225b29d7ba367b6111ab3351e54cc7c01fa7646a7a12712"
    ),
    "benchmarks/livecodebench_full_release_v6_2026_05_22/"
    "full_n8_public_select_summary.json": (
        "2a0ff919aa15eb9ecdf74824f7bf790a23f6d0197ef74970b6190c60e0e00772"
    ),
    "benchmarks/evalplus_l20_codeforge_2026_05_22/summary.csv": (
        "08732bbb76450f92ef3c02fa97a163aba01f71028365072c205c5a3af45d5550"
    ),
    "benchmarks/livecodebench_full_release_v6_2026_05_22/"
    "qwen25_coder_7b_temp08_n8_public_select_full_eval/report.json": (
        "7272f5591c2f868c059226a2a5ec8fc772994cfafd20eb8397a2b6d90aed64bf"
    ),
    "benchmarks/evalplus_l20_codeforge_2026_05_22/rechecks/manifest.json": (
        "e86db2af864a9c8896dcd1bc2d4d7b44af7fa395b856ea02b6f0e69c31c915cc"
    ),
}


@dataclass(frozen=True)
class ArtifactVerification:
    path: str
    expected_sha256: str
    actual_sha256: str | None
    status: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(
    root: Path = Path("."),
    expected_hashes: Mapping[str, str] = DEFAULT_ARTIFACT_HASHES,
) -> list[ArtifactVerification]:
    """Verify claim-bearing artifacts without reading outside ``root``."""
    resolved_root = root.resolve()
    results: list[ArtifactVerification] = []
    for relative_path, expected_sha256 in expected_hashes.items():
        candidate = (resolved_root / relative_path).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            raise ValueError(f"artifact path escapes root: {relative_path}") from None

        if not candidate.is_file():
            results.append(
                ArtifactVerification(relative_path, expected_sha256, None, "missing")
            )
            continue
        actual_sha256 = sha256_file(candidate)
        status = "ok" if actual_sha256 == expected_sha256 else "mismatch"
        results.append(
            ArtifactVerification(relative_path, expected_sha256, actual_sha256, status)
        )
    return results


def artifact_report(
    root: Path = Path("."),
    expected_hashes: Mapping[str, str] = DEFAULT_ARTIFACT_HASHES,
) -> dict[str, object]:
    results = verify_artifacts(root=root, expected_hashes=expected_hashes)
    return {
        "status": "PASS" if all(result.status == "ok" for result in results) else "FAIL",
        "artifacts": [asdict(result) for result in results],
    }
