#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
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


REPAIR = load_script_module("repair_lcb_generations", "repair_lcb_generations.py")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def syntax_error_message(code: str) -> str | None:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"{exc.msg} at line {exc.lineno}"
    return None


def records_by_question(path: Path, key: str = "question_id") -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = load_json(path)
    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    else:
        records = payload
    if not isinstance(records, list):
        return {}
    return {
        str(record[key]): record
        for record in records
        if isinstance(record, dict) and record.get(key) is not None
    }


def public_records_by_question(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = load_json(path)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return {}
    return {
        str(record["question_id"]): record
        for record in records
        if isinstance(record, dict) and record.get("question_id") is not None
    }


def eval_records_by_question(path: Path) -> dict[str, dict[str, Any]]:
    return records_by_question(path)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    generations_path = run_dir / "generations.json"
    public_selection_path = run_dir / "public_selection.json"
    eval_all_path = run_dir / "eval_all.json"
    if not public_selection_path.exists() and (run_dir / "eval" / "public_selection.json").exists():
        public_selection_path = run_dir / "eval" / "public_selection.json"
    if not eval_all_path.exists() and (run_dir / "eval" / "eval_all.json").exists():
        eval_all_path = run_dir / "eval" / "eval_all.json"

    generation_records = records_by_question(generations_path)
    public_records = public_records_by_question(public_selection_path)
    eval_records = eval_records_by_question(eval_all_path)
    question_ids = sorted(set(generation_records) | set(public_records) | set(eval_records))

    records: list[dict[str, Any]] = []
    totals = {
        "tasks": 0,
        "candidates": 0,
        "syntax_ok_candidates": 0,
        "entrypoint_candidates": 0,
        "public_oracle_pass_tasks": 0,
        "public_selected_pass_tasks": 0,
        "hidden_selected_pass_tasks": 0,
        "selected_syntax_error_tasks": 0,
    }
    for question_id in question_ids:
        generation_record = generation_records.get(question_id, {})
        public_record = public_records.get(question_id, {})
        eval_record = eval_records.get(question_id, {})
        code_list = [
            item for item in generation_record.get("code_list", []) if isinstance(item, str)
        ]
        candidate_health = []
        for index, code in enumerate(code_list):
            syntax_error = syntax_error_message(code)
            has_entrypoint = REPAIR.has_entrypoint(code)
            candidate_health.append(
                {
                    "index": index,
                    "length": len(code),
                    "syntax_ok": syntax_error is None,
                    "syntax_error": syntax_error,
                    "has_entrypoint": has_entrypoint,
                }
            )

        public_scores = [float(score) for score in public_record.get("public_scores", [])]
        selected_index = public_record.get("selected_index")
        selected_public_score = (
            public_scores[int(selected_index)]
            if isinstance(selected_index, int) and selected_index < len(public_scores)
            else public_record.get("selected_public_score")
        )
        graded_list = eval_record.get("graded_list") or []
        hidden_selected_pass = bool(graded_list[0]) if graded_list else None
        metadata = eval_record.get("metadata") or []
        selected_metadata = metadata[0] if metadata and isinstance(metadata[0], dict) else {}
        selected_health = (
            candidate_health[int(selected_index)]
            if isinstance(selected_index, int) and selected_index < len(candidate_health)
            else None
        )

        syntax_ok_count = sum(item["syntax_ok"] for item in candidate_health)
        entrypoint_count = sum(item["has_entrypoint"] for item in candidate_health)
        public_oracle_pass = bool(public_record.get("public_oracle_pass"))
        public_selected_pass = selected_public_score == 1.0
        selected_syntax_error = bool(
            selected_health is not None and not selected_health["syntax_ok"]
        )

        totals["tasks"] += 1
        totals["candidates"] += len(candidate_health)
        totals["syntax_ok_candidates"] += syntax_ok_count
        totals["entrypoint_candidates"] += entrypoint_count
        totals["public_oracle_pass_tasks"] += int(public_oracle_pass)
        totals["public_selected_pass_tasks"] += int(public_selected_pass)
        totals["hidden_selected_pass_tasks"] += int(hidden_selected_pass is True)
        totals["selected_syntax_error_tasks"] += int(selected_syntax_error)

        if public_oracle_pass:
            failure_mode = "has_public_passing_candidate"
        elif syntax_ok_count == 0:
            failure_mode = "no_syntax_valid_candidates"
        elif entrypoint_count == 0:
            failure_mode = "no_entrypoint_candidates"
        elif selected_syntax_error:
            failure_mode = "selector_chose_syntax_invalid_candidate"
        else:
            failure_mode = "public_tests_reject_syntax_valid_candidates"

        records.append(
            {
                "question_id": question_id,
                "question_title": generation_record.get("question_title")
                or public_record.get("question_title"),
                "difficulty": generation_record.get("difficulty"),
                "n_candidates": len(candidate_health),
                "syntax_ok_candidates": syntax_ok_count,
                "entrypoint_candidates": entrypoint_count,
                "public_scores": public_scores,
                "public_oracle_pass": public_oracle_pass,
                "selected_index": selected_index,
                "selected_public_score": selected_public_score,
                "hidden_selected_pass": hidden_selected_pass,
                "selected_error_code": selected_metadata.get("error_code"),
                "selected_error_message": selected_metadata.get("error_message"),
                "failure_mode": failure_mode,
                "candidate_health": candidate_health,
            }
        )

    rates = {
        "syntax_ok_candidate_rate": (
            totals["syntax_ok_candidates"] / totals["candidates"]
            if totals["candidates"]
            else 0.0
        ),
        "entrypoint_candidate_rate": (
            totals["entrypoint_candidates"] / totals["candidates"]
            if totals["candidates"]
            else 0.0
        ),
        "public_oracle_task_rate": (
            totals["public_oracle_pass_tasks"] / totals["tasks"] if totals["tasks"] else 0.0
        ),
        "hidden_selected_task_rate": (
            totals["hidden_selected_pass_tasks"] / totals["tasks"] if totals["tasks"] else 0.0
        ),
    }
    return {
        "run_dir": str(run_dir),
        "generations": str(generations_path) if generations_path.exists() else None,
        "public_selection": str(public_selection_path) if public_selection_path.exists() else None,
        "eval_all": str(eval_all_path) if eval_all_path.exists() else None,
        "totals": totals,
        "rates": rates,
        "records": records,
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    runs = [summarize_run(Path(run_dir)) for run_dir in args.run_dir]
    totals = {
        "runs": len(runs),
        "tasks": sum(run["totals"]["tasks"] for run in runs),
        "candidates": sum(run["totals"]["candidates"] for run in runs),
        "syntax_ok_candidates": sum(run["totals"]["syntax_ok_candidates"] for run in runs),
        "entrypoint_candidates": sum(run["totals"]["entrypoint_candidates"] for run in runs),
        "public_oracle_pass_tasks": sum(
            run["totals"]["public_oracle_pass_tasks"] for run in runs
        ),
        "hidden_selected_pass_tasks": sum(
            run["totals"]["hidden_selected_pass_tasks"] for run in runs
        ),
        "selected_syntax_error_tasks": sum(
            run["totals"]["selected_syntax_error_tasks"] for run in runs
        ),
    }
    payload = {
        "metadata": {
            "source": "lcb_candidate_health_audit_v1",
            "hidden_expected_outputs_used": False,
        },
        "totals": totals,
        "rates": {
            "syntax_ok_candidate_rate": (
                totals["syntax_ok_candidates"] / totals["candidates"]
                if totals["candidates"]
                else 0.0
            ),
            "entrypoint_candidate_rate": (
                totals["entrypoint_candidates"] / totals["candidates"]
                if totals["candidates"]
                else 0.0
            ),
            "public_oracle_task_rate": (
                totals["public_oracle_pass_tasks"] / totals["tasks"]
                if totals["tasks"]
                else 0.0
            ),
            "hidden_selected_task_rate": (
                totals["hidden_selected_pass_tasks"] / totals["tasks"]
                if totals["tasks"]
                else 0.0
            ),
        },
        "runs": runs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "totals": totals, "rates": payload["rates"]}, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit saved LCB generation runs for candidate syntax, entrypoints, and public-selection health."
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    build_audit(build_parser().parse_args())


if __name__ == "__main__":
    main()
