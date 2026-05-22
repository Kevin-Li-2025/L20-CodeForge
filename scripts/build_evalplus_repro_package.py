from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    dataset: str
    protocol: str
    sample_file: str | None
    report_file: str
    notes: str


CASES = [
    BenchmarkCase(
        name="humaneval_greedy",
        dataset="humaneval",
        protocol="greedy_model",
        sample_file="humaneval.greedy.samples.jsonl",
        report_file="humaneval.greedy.evalplus_report.json",
        notes="Greedy n=1 model baseline, official EvalPlus scoring.",
    ),
    BenchmarkCase(
        name="humaneval_n10_sampling",
        dataset="humaneval",
        protocol="sampled_pass_at_k",
        sample_file="humaneval.temp08.n10.samples.jsonl",
        report_file="humaneval.temp08.n10.evalplus_report.json",
        notes="Fixed n=10 sampling report; pass@10 is reported by official EvalPlus.",
    ),
    BenchmarkCase(
        name="humaneval_clean_system_best",
        dataset="humaneval",
        protocol="clean_public_signal_system",
        sample_file="humaneval.mixed-target.literal-combined.public-consensus-selected.samples.jsonl",
        report_file="humaneval.mixed-target.literal-combined.public-consensus-selected.evalplus_report.json",
        notes="Best non-symbolic HumanEval+ system result; selection does not read extra tests.",
    ),
    BenchmarkCase(
        name="humaneval_symbolic_ablation",
        dataset="humaneval",
        protocol="task_specific_symbolic_ablation",
        sample_file="humaneval.mixed-target.literal-symbolic-combined.base-longest-selected.samples.jsonl",
        report_file="humaneval.mixed-target.literal-symbolic-combined.base-longest-selected.evalplus_report.json",
        notes="Transparent task-specific ablation, not a general model/system score.",
    ),
    BenchmarkCase(
        name="mbpp_greedy",
        dataset="mbpp",
        protocol="greedy_model",
        sample_file="mbpp.greedy.samples.jsonl",
        report_file="mbpp.greedy.evalplus_report.json",
        notes="Greedy n=1 model baseline, official EvalPlus scoring.",
    ),
    BenchmarkCase(
        name="mbpp_clean_system_best",
        dataset="mbpp",
        protocol="clean_public_signal_system",
        sample_file="mbpp.temp08.n5-plus-basefallback-n30.public-consensus-shortest-selected.samples.jsonl",
        report_file="mbpp.temp08.n5-plus-basefallback-n30.public-consensus-shortest-selected.evalplus_report.json",
        notes="Best clean MBPP+ system result: public-base fallback resampling plus public-input consensus-shortest selection.",
    ),
    BenchmarkCase(
        name="mbpp_n5_selector",
        dataset="mbpp",
        protocol="clean_public_signal_system_ablation",
        sample_file="mbpp.temp08.n5.base-longest-selected.samples.jsonl",
        report_file="mbpp.temp08.n5.base-longest-selected.evalplus_report.json",
        notes="First MBPP n=5 base-test selector ablation.",
    ),
    BenchmarkCase(
        name="mbpp_consensus_shortest",
        dataset="mbpp",
        protocol="clean_public_signal_system_ablation",
        sample_file="mbpp.temp08.n5.public-consensus-shortest-selected.samples.jsonl",
        report_file="mbpp.temp08.n5.public-consensus-shortest-selected.evalplus_report.json",
        notes="MBPP n=5 public-input consensus with shortest tie-breaker.",
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_scores(report_payload: dict[str, Any]) -> dict[str, float]:
    scores = report_payload.get("scores")
    if isinstance(scores, dict) and scores:
        return {str(key): float(value) for key, value in scores.items() if value is not None}
    parsed = parse_evalplus_stdout_scores(str(report_payload.get("stdout", "")))
    if parsed:
        return parsed
    fallback: dict[str, float] = {}
    base = report_payload.get("base_pass_at_1")
    plus = report_payload.get("plus_pass_at_1")
    if base is not None:
        fallback["base_pass@1"] = float(base)
    if plus is not None:
        fallback["plus_pass@1"] = float(plus)
    return fallback


def parse_evalplus_stdout_scores(stdout: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    current: str | None = None
    lines = [line.strip() for line in stdout.splitlines()]
    for index, line in enumerate(lines):
        if line.endswith("(base tests)"):
            current = "base"
            continue
        if line.endswith("(base + extra tests)"):
            current = "plus"
            continue
        if current and line.startswith("pass@"):
            metric, _, value = line.partition(":")
            value = value.strip()
            if not value and index + 1 < len(lines):
                value = lines[index + 1].strip()
            try:
                parsed[f"{current}_{metric}"] = float(value)
            except ValueError:
                continue
    return parsed


def copy_artifact(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def build_package(
    artifact_root: Path,
    output_dir: Path,
    source_commit: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    reports_dir = output_dir / "reports"
    rows: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []

    for case in CASES:
        report_source = artifact_root / case.report_file
        if not report_source.exists():
            raise FileNotFoundError(report_source)
        report_payload = read_report(report_source)
        report_copy = copy_artifact(report_source, reports_dir / case.report_file)
        sample_copy = None
        if case.sample_file is not None:
            sample_source = artifact_root / case.sample_file
            if not sample_source.exists():
                raise FileNotFoundError(sample_source)
            sample_copy = copy_artifact(sample_source, samples_dir / case.sample_file)

        scores = extract_scores(report_payload)
        row = {
            "name": case.name,
            "dataset": case.dataset,
            "protocol": case.protocol,
            "base_pass@1": scores.get("base_pass@1", report_payload.get("base_pass_at_1")),
            "plus_pass@1": scores.get("plus_pass@1", report_payload.get("plus_pass_at_1")),
            "base_pass@10": scores.get("base_pass@10"),
            "plus_pass@10": scores.get("plus_pass@10"),
            "sample_file": case.sample_file or "",
            "report_file": case.report_file,
            "notes": case.notes,
        }
        rows.append(row)
        manifest_cases.append(
            {
                **row,
                "report_artifact": report_copy,
                "sample_artifact": sample_copy,
                "source_report": str(report_source),
                "source_sample": str(artifact_root / case.sample_file)
                if case.sample_file is not None
                else None,
            }
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "package": output_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": source_commit or git_commit(),
        "artifact_root": str(artifact_root),
        "evalplus_version_note": (
            "Use the installed evalplus package and official evalplus.evaluate command; "
            "the copied reports preserve original stdout/stderr."
        ),
        "cases": manifest_cases,
        "summary_artifact": {
            "path": str(summary_path),
            "size_bytes": summary_path.stat().st_size,
            "sha256": sha256_file(summary_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    readme_path = output_dir / "README.md"
    readme_path.write_text(render_readme(rows), encoding="utf-8")
    return manifest


def render_readme(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# EvalPlus Reproducibility Package",
        "",
        "This package contains the selected sample files, official EvalPlus report JSONs,",
        "a machine-readable manifest with SHA256 hashes, and a CSV summary for the",
        "L20-CodeForge public benchmark sprint.",
        "",
        "## Protocol Boundary",
        "",
        "- Greedy rows are model baselines.",
        "- Clean system rows may use public base tests, public prompt/base inputs,",
        "  multi-sample generation, targeted resampling, and deterministic selection.",
        "- Clean system rows do not use EvalPlus extra tests for candidate selection.",
        "- The symbolic ablation row is transparent and task-specific; do not present it",
        "  as a general model or clean system score.",
        "",
        "## Summary",
        "",
        "| name | dataset | protocol | base pass@1 | plus pass@1 | base pass@10 | plus pass@10 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {dataset} | {protocol} | {base} | {plus} | {base10} | {plus10} |".format(
                name=row["name"],
                dataset=row["dataset"],
                protocol=row["protocol"],
                base=format_score(row["base_pass@1"]),
                plus=format_score(row["plus_pass@1"]),
                base10=format_score(row["base_pass@10"]),
                plus10=format_score(row["plus_pass@10"]),
            )
        )
    lines.extend(
        [
            "",
            "## Re-run Official Scoring",
            "",
            "From the repository root on a machine with `l20_codeforge[bench]` installed:",
            "",
            "```bash",
            "python -m l20_codeforge eval-evalplus mbpp \\",
            "  benchmarks/evalplus_l20_codeforge_2026_05_22/samples/mbpp.temp08.n5-plus-basefallback-n30.public-consensus-shortest-selected.samples.jsonl \\",
            "  --output /tmp/mbpp_recheck.json \\",
            "  --parallel 8",
            "",
            "python -m l20_codeforge eval-evalplus humaneval \\",
            "  benchmarks/evalplus_l20_codeforge_2026_05_22/samples/humaneval.mixed-target.literal-combined.public-consensus-selected.samples.jsonl \\",
            "  --output /tmp/humaneval_recheck.json \\",
            "  --parallel 8",
            "```",
            "",
            "Check `manifest.json` before rerunning to verify file hashes.",
            "",
            "## Rechecks",
            "",
            "If official rechecks were run after packaging, their reports should be",
            "stored under `rechecks/` with hashes in `rechecks/manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def format_score(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    manifest = build_package(args.artifact_root, args.output_dir, source_commit=args.source_commit)
    print(json.dumps({"output_dir": str(args.output_dir), "cases": len(manifest["cases"])}, indent=2))


if __name__ == "__main__":
    main()
