#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any


def load_script_module(name: str, script_name: str) -> Any:
    script = Path(__file__).with_name(script_name)
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LCB_RUNNER = load_script_module("run_lcb_subset_benchmark", "run_lcb_subset_benchmark.py")

TOP_LEVEL_SOLUTION_METHOD = re.compile(r"^def\s+\w+\s*\(\s*self(?:\s*,|\s*\))")
TAIL_MARKER = re.compile(
    r"^(?:```|#{2,}\s|Explanation\b|Approach\b|Complexity\b|Note\b|The solution\b)",
    flags=re.IGNORECASE,
)


def parse_question_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    question_ids = {item.strip() for item in value.split(",") if item.strip()}
    if not question_ids:
        raise ValueError("--question-ids must contain at least one id")
    return question_ids


def has_entrypoint(code: str) -> bool:
    return (
        "class Solution" in code
        or "input(" in code
        or "sys.stdin" in code
        or 'if __name__ == "__main__"' in code
        or "if __name__ == '__main__'" in code
    )


def strip_markdown_tail(code: str) -> str:
    lines = code.strip().splitlines()
    kept: list[str] = []
    seen_entrypoint = False
    for line in lines:
        if "class Solution" in line or TOP_LEVEL_SOLUTION_METHOD.match(line):
            seen_entrypoint = True
        if seen_entrypoint and line and not line[0].isspace() and TAIL_MARKER.match(line.strip()):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def extract_definition_region(code: str) -> str:
    stripped = code.strip()
    class_index = stripped.find("class Solution")
    if class_index >= 0:
        return stripped[class_index:].strip()

    lines = stripped.splitlines()
    for index, line in enumerate(lines):
        if TOP_LEVEL_SOLUTION_METHOD.match(line):
            method_block = "\n".join(lines[index:]).strip()
            return "class Solution:\n" + "\n".join(
                f"    {method_line}" if method_line else ""
                for method_line in method_block.splitlines()
            )
    return stripped


def trim_to_compilable_prefix(code: str) -> str:
    stripped = code.strip()
    if not stripped:
        return stripped
    try:
        ast.parse(stripped)
        return stripped
    except SyntaxError as exc:
        syntax_error_lineno = exc.lineno
        if not syntax_error_lineno:
            return stripped

    lines = stripped.splitlines()
    for end in range(min(syntax_error_lineno - 1, len(lines)), 0, -1):
        candidate = "\n".join(lines[:end]).strip()
        if not candidate or not has_entrypoint(candidate):
            continue
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        return candidate
    return stripped


def repair_code_text(text: str) -> str:
    code = LCB_RUNNER.strip_lcb_code_block(text)
    code = extract_definition_region(code)
    code = strip_markdown_tail(code)
    return trim_to_compilable_prefix(code)


def syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def repair_record(record: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_outputs = [item for item in record.get("raw_outputs", []) if isinstance(item, str)]
    old_code_list = [item for item in record.get("code_list", []) if isinstance(item, str)]
    sample_count = max(len(raw_outputs), len(old_code_list))
    repaired_codes: list[str] = []
    candidate_reports: list[dict[str, Any]] = []
    for index in range(sample_count):
        source = raw_outputs[index] if index < len(raw_outputs) else old_code_list[index]
        old_code = old_code_list[index] if index < len(old_code_list) else ""
        repaired = repair_code_text(source)
        if not repaired.strip() and old_code:
            repaired = repair_code_text(old_code) or old_code
        repaired_codes.append(repaired)
        candidate_reports.append(
            {
                "candidate_index": index,
                "old_len": len(old_code),
                "new_len": len(repaired),
                "changed": old_code != repaired,
                "syntax_ok": syntax_ok(repaired),
                "has_entrypoint": has_entrypoint(repaired),
            }
        )

    repaired_record = dict(record)
    repaired_record["code_list"] = repaired_codes
    repaired_record["repair"] = {
        "source": "raw_outputs" if raw_outputs else "code_list",
        "version": "extract_fenced_after_think_v1",
        "changed_candidates": sum(item["changed"] for item in candidate_reports),
        "syntax_ok_candidates": sum(item["syntax_ok"] for item in candidate_reports),
        "entrypoint_candidates": sum(item["has_entrypoint"] for item in candidate_reports),
    }
    return repaired_record, candidate_reports


def repair_generations(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_suffix(".repair.json")
    question_ids = parse_question_ids(args.question_ids)
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("input generations must be a JSON list")

    repaired_records: list[dict[str, Any]] = []
    repair_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        question_id = str(record.get("question_id", ""))
        if question_ids is not None and question_id not in question_ids:
            repaired_records.append(record)
            continue
        repaired_record, candidate_reports = repair_record(record)
        repaired_records.append(repaired_record)
        repair_records.append(
            {
                "question_id": question_id,
                "question_title": record.get("question_title"),
                "difficulty": record.get("difficulty"),
                "candidates": candidate_reports,
                "changed_candidates": sum(item["changed"] for item in candidate_reports),
                "syntax_ok_candidates": sum(item["syntax_ok"] for item in candidate_reports),
                "entrypoint_candidates": sum(item["has_entrypoint"] for item in candidate_reports),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(repaired_records, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "records_seen": len(records),
        "records_repaired": len(repair_records),
        "question_ids": sorted(question_ids) if question_ids is not None else None,
        "changed_candidates": sum(item["changed_candidates"] for item in repair_records),
        "syntax_ok_candidates": sum(item["syntax_ok_candidates"] for item in repair_records),
        "entrypoint_candidates": sum(item["entrypoint_candidates"] for item in repair_records),
        "seconds": round(time.monotonic() - started, 3),
        "records": repair_records,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-extract clean LiveCodeBench code candidates from saved raw model outputs. "
            "This is deterministic post-processing for interrupted or reasoning-heavy outputs."
        )
    )
    parser.add_argument("--input", required=True, help="Input generations.json")
    parser.add_argument("--output", required=True, help="Output repaired generations.json")
    parser.add_argument("--report", help="Output repair report JSON")
    parser.add_argument("--question-ids", help="Comma-separated question ids to repair")
    return parser


def main() -> None:
    repair_generations(build_parser().parse_args())


if __name__ == "__main__":
    main()
