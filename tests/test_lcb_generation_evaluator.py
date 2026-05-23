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


def test_evaluator_applies_saved_public_selection_by_question_id() -> None:
    evaluator = load_evaluator_module()
    problem_a = type("Problem", (), {"question_id": "a"})()
    problem_b = type("Problem", (), {"question_id": "b"})()

    selected, raw_outputs, records = evaluator.apply_public_selection_records(
        problems=[problem_a, problem_b],
        generations=[["a0", "a1"], ["b0", "b1"]],
        raw_outputs=[["ra0", "ra1"], ["rb0", "rb1"]],
        selection_records=[
            {"question_id": "b", "selected_index": 0},
            {"question_id": "a", "selected_index": 1},
        ],
    )

    assert selected == [["a1"], ["b0"]]
    assert raw_outputs == [["ra1"], ["rb0"]]
    assert [record["question_id"] for record in records] == ["a", "b"]


def test_evaluator_behavior_selection_uses_consensus_after_public_score() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_behavior_selected_index(
        public_results=[[True], [True], [True]],
        code_outputs=["long but okay", "short", "tiny"],
        behavior_outputs=[
            ["OK:3:aaa", "OK:3:bbb"],
            ["OK:3:aaa", "OK:3:bbb"],
            ["OK:3:ccc", "OK:3:ddd"],
        ],
        tie_breaker="shortest",
    )

    assert selected == 1


def test_evaluator_behavior_selection_keeps_public_score_primary() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_behavior_selected_index(
        public_results=[[True], [False], [False]],
        code_outputs=["public-pass", "wrong-a", "wrong-b"],
        behavior_outputs=[
            ["OK:3:aaa"],
            ["OK:3:bbb"],
            ["OK:3:bbb"],
        ],
        tie_breaker="shortest",
    )

    assert selected == 0


def test_evaluator_behavior_selection_can_reuse_public_scores() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_behavior_selected_index_from_scores(
        public_scores=[1.0, 1.0, 0.5],
        code_outputs=["long public pass", "short", "almost"],
        behavior_outputs=[
            ["OK:3:aaa"],
            ["OK:3:aaa"],
            ["OK:3:aaa"],
        ],
        tie_breaker="shortest",
    )

    assert selected == 1


def test_evaluator_loads_external_behavior_inputs(tmp_path: Path) -> None:
    evaluator = load_evaluator_module()
    payload = {
        "records": [
            {"question_id": "a", "inputs": ["1\n", "", 4]},
            {"question_id": "b", "inputs": ["[1]\n2"]},
            {"question_id": "c", "inputs": "bad"},
        ]
    }
    path = tmp_path / "behavior_inputs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = evaluator.load_behavior_input_payload(path)

    assert loaded == {"a": ["1\n"], "b": ["[1]\n2"]}


def test_evaluator_reconstructs_public_scores_by_question_id() -> None:
    evaluator = load_evaluator_module()
    problem_a = type("Problem", (), {"question_id": "a"})()
    problem_b = type("Problem", (), {"question_id": "b"})()

    scores = evaluator.public_scores_from_selection_records(
        problems=[problem_a, problem_b],
        generations=[["a0", "a1"], ["b0"]],
        selection_records=[
            {"question_id": "b", "public_scores": [0.25, 1.0]},
            {"question_id": "a", "public_scores": [1.0, 0.0]},
        ],
    )

    assert scores == {0: [1.0, 0.0], 1: [0.25]}


def test_evaluator_generates_deterministic_behavior_mutations() -> None:
    evaluator = load_evaluator_module()

    functional = evaluator.mutate_functional_input("[1, 2, 3]\n\"abc\"", limit=4)
    stdin = evaluator.mutate_stdin_input("2\n3 4\n5 6\n", limit=4)

    assert functional == evaluator.mutate_functional_input("[1, 2, 3]\n\"abc\"", limit=4)
    assert stdin == evaluator.mutate_stdin_input("2\n3 4\n5 6\n", limit=4)
    assert "[1, 2, 3]\n\"abc\"" not in functional
    assert "2\n3 4\n5 6\n" not in stdin


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
