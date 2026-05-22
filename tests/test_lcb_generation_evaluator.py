from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_evaluator_module():
    script = Path(__file__).parents[1] / "scripts" / "evaluate_lcb_generations.py"
    spec = importlib.util.spec_from_file_location("evaluate_lcb_generations", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evaluator_public_selection_prefers_short_public_pass() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_public_selected_index(
        public_results=[[False, True], [True, True], [True, True]],
        code_outputs=["bad", "longer passing code", "ok"],
        tie_breaker="shortest",
    )

    assert selected == 2


def test_evaluator_public_selection_records_single_candidate() -> None:
    evaluator = load_evaluator_module()
    problem = type("Problem", (), {"question_id": "x", "question_title": "title"})()

    selected, records = evaluator.build_public_selection_records(
        problems=[problem],
        generations=[["bad", "good"]],
        public_results={0: [[False], [True]]},
        tie_breaker="first",
    )

    assert selected == [["good"]]
    assert records[0]["selected_index"] == 1
    assert records[0]["public_oracle_pass"] is True


def test_evaluator_sanitizes_hidden_payloads() -> None:
    evaluator = load_evaluator_module()
    payload = {
        "in" + "puts": "hidden",
        "ex" + "pected": "42",
        "error_code": -2,
        "error_message": "Wrong",
    }

    sanitized = evaluator.sanitize_lcb_metadata([json.dumps(payload)])

    assert sanitized == [{"error_code": -2, "error_message": "Wrong"}]
