from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_comparison_module():
    path = Path(__file__).parents[1] / "scripts" / "compare_code_stage_rollouts.py"
    spec = importlib.util.spec_from_file_location("compare_code_stage_rollouts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_stage(path: Path, outcomes: list[bool]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": f"task-{index}",
                    "sample_index": 0,
                    "all_passed": passed,
                }
            )
            + "\n"
            for index, passed in enumerate(outcomes)
        ),
        encoding="utf-8",
    )


def test_build_report_computes_paired_stage_deltas(tmp_path: Path) -> None:
    module = load_comparison_module()
    base = tmp_path / "base.jsonl"
    sft = tmp_path / "sft.jsonl"
    rlvr = tmp_path / "rlvr.jsonl"
    write_stage(base, [True, False, False, True])
    write_stage(sft, [True, True, False, False])
    write_stage(rlvr, [True, True, True, True])

    report = module.build_report(
        {"base": base, "sft": sft, "rlvr": rlvr},
        expected_tasks=4,
        required_base_to_final_delta=0.25,
    )

    assert report["stages"]["base"]["greedy_accuracy"] == 0.5
    assert report["stages"]["rlvr"]["greedy_accuracy"] == 1.0
    assert report["paired_comparisons"]["base_to_sft"]["net_tasks"] == 0
    assert report["paired_comparisons"]["base_to_rlvr"]["gained_tasks"] == 2
    assert report["development_gate"]["passed"] is True
