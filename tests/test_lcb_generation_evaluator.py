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


def load_behavior_generator_module():
    script = Path(__file__).parents[1] / "scripts" / "generate_lcb_behavior_tests.py"
    spec = importlib.util.spec_from_file_location("generate_lcb_behavior_tests", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_runner_module():
    script = Path(__file__).parents[1] / "scripts" / "run_lcb_subset_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_lcb_subset_benchmark", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_adaptive_differential_module():
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "build_lcb_adaptive_differential_inputs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_lcb_adaptive_differential_inputs",
        script,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_expected_verifier_module():
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "build_lcb_expected_output_verifier_prompts.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_lcb_expected_output_verifier_prompts",
        script,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_expected_selector_module():
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "select_lcb_expected_verifier_candidates.py"
    )
    spec = importlib.util.spec_from_file_location(
        "select_lcb_expected_verifier_candidates",
        script,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_chunk_evaluator_module():
    script = Path(__file__).parents[1] / "scripts" / "evaluate_lcb_selection_chunks.py"
    spec = importlib.util.spec_from_file_location("evaluate_lcb_selection_chunks", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_public_selection_builder_module():
    script = Path(__file__).parents[1] / "scripts" / "build_lcb_public_selection.py"
    spec = importlib.util.spec_from_file_location("build_lcb_public_selection", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_repair_module():
    script = Path(__file__).parents[1] / "scripts" / "repair_lcb_generations.py"
    spec = importlib.util.spec_from_file_location("repair_lcb_generations", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_candidate_health_audit_module():
    script = Path(__file__).parents[1] / "scripts" / "audit_lcb_candidate_health.py"
    spec = importlib.util.spec_from_file_location("audit_lcb_candidate_health", script)
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


def test_evaluator_public_tie_breaker_prefers_static_health() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_public_selected_index_from_scores(
        public_scores=[0.0, 0.0],
        code_outputs=[
            "class Solution:\n    def answer(self,",
            "class Solution:\n    def answer(self):\n        return 1",
        ],
        tie_breaker="shortest",
    )

    assert selected == 1


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


def test_chunk_evaluator_metrics_from_pass_lists() -> None:
    chunk_evaluator = load_chunk_evaluator_module()

    metrics = chunk_evaluator.metrics_from_pass_lists([[True], [False], []])

    assert metrics["pass@1"] == 1 / 3
    assert metrics["detail"]["pass@1"] == {"0": 1.0, "1": 0.0, "2": 0.0}


def test_chunk_evaluator_remaps_results_by_chunk_start() -> None:
    chunk_evaluator = load_chunk_evaluator_module()

    results, metadata, pass_lists, eval_all = chunk_evaluator.remap_results(
        [
            {
                "start_index": 2,
                "results": {"0": [[False]], "1": [[True, True]]},
                "metadata": [[{"error_code": "x"}], [{}]],
                "pass_lists": [[False], [True]],
                "eval_all": [{"question_id": "c"}, {"question_id": "d"}],
            },
            {
                "start_index": 0,
                "results": {"0": [[True]], "1": [[False]]},
                "metadata": [[{}], [{}]],
                "pass_lists": [[True], [False]],
                "eval_all": [{"question_id": "a"}, {"question_id": "b"}],
            },
        ]
    )

    assert results == {0: [[True]], 1: [[False]], 2: [[False]], 3: [[True, True]]}
    assert pass_lists == [[True], [False], [False], [True]]
    assert [item[0].get("error_code") for item in metadata] == [None, None, "x", None]
    assert [item["question_id"] for item in eval_all] == ["a", "b", "c", "d"]


def test_public_selection_builder_aligns_saved_generations() -> None:
    builder = load_public_selection_builder_module()
    problem_a = type("Problem", (), {"question_id": "a"})()
    problem_b = type("Problem", (), {"question_id": "b"})()
    problem_c = type("Problem", (), {"question_id": "c"})()

    problems, generations = builder.select_problem_generations(
        problems=[problem_a, problem_b, problem_c],
        generation_records={
            "b": {"code_list": ["b0", "b1"]},
            "c": {"code_list": []},
        },
        max_samples=1,
    )

    assert [problem.question_id for problem in problems] == ["b"]
    assert generations == [["b0"]]


def test_public_selection_builder_uses_max_candidate_for_k_list() -> None:
    builder = load_public_selection_builder_module()

    assert builder.public_k_list_for_generations([["a"], ["b", "c", "d"]]) == [1, 3]


def test_lcb_runner_raw_prompt_rendering_does_not_apply_chat_template() -> None:
    runner = load_runner_module()
    tokenizer = object()

    rendered = runner.render_model_input(
        tokenizer=tokenizer,
        prompt="### Question\nSolve it",
        prompt_rendering="raw",
        system_message="system",
    )

    assert rendered == "### Question\nSolve it"


def test_lcb_runner_strip_extracts_answer_tag() -> None:
    runner = load_runner_module()

    code = runner.strip_lcb_code_block("<think>reason</think><answer>print(1)</answer>")

    assert code == "print(1)"


def test_lcb_runner_strip_extracts_fenced_answer_tag() -> None:
    runner = load_runner_module()

    code = runner.strip_lcb_code_block(
        "<answer>Explanation\n```python\nprint(1)\n```\n</answer>"
    )

    assert code == "print(1)"


def test_lcb_runner_strip_extracts_after_think_close() -> None:
    runner = load_runner_module()

    code = runner.strip_lcb_code_block("<think>reason</think>\nprint(2)")

    assert code == "print(2)"


def test_lcb_runner_strip_extracts_fenced_code_after_think_close() -> None:
    runner = load_runner_module()

    code = runner.strip_lcb_code_block(
        "<think>reason</think>\n"
        "Here is the final code.\n"
        "```python\n"
        "class Solution:\n"
        "    def solve(self):\n"
        "        return 1\n"
        "```\n"
    )

    assert code == "class Solution:\n    def solve(self):\n        return 1"


def test_repair_generation_extracts_fenced_code_from_raw_output() -> None:
    repair = load_repair_module()

    code = repair.repair_code_text(
        "<think>reason</think>\n"
        "Use this code.\n"
        "```python\n"
        "class Solution:\n"
        "    def answer(self):\n"
        "        return 42\n"
        "```\n"
    )

    assert code == "class Solution:\n    def answer(self):\n        return 42"


def test_repair_generation_wraps_top_level_solution_method() -> None:
    repair = load_repair_module()

    code = repair.repair_code_text(
        "# Your code here\n"
        "def smallestString(self, s: str) -> str:\n"
        "    return s\n"
    )

    assert code == "class Solution:\n    def smallestString(self, s: str) -> str:\n        return s"


def test_repair_generation_trims_markdown_tail() -> None:
    repair = load_repair_module()

    code = repair.repair_code_text(
        "```python\n"
        "class Solution:\n"
        "    def answer(self):\n"
        "        return 42\n"
        "```\n"
        "\n"
        "### Explanation\n"
        "This is not code.\n"
    )

    assert code == "class Solution:\n    def answer(self):\n        return 42"


def test_repair_generation_trims_incomplete_syntax_tail() -> None:
    repair = load_repair_module()

    code = repair.repair_code_text(
        "```python\n"
        "class Solution:\n"
        "    def answer(self):\n"
        "        return 42\n"
        "    def broken(self,\n"
    )

    assert code == "class Solution:\n    def answer(self):\n        return 42"


def test_repair_record_keeps_old_code_when_raw_extraction_is_empty() -> None:
    repair = load_repair_module()

    repaired, reports = repair.repair_record(
        {
            "question_id": "x",
            "raw_outputs": ["<think>reason</think>\n### Solution Code\n```"],
            "code_list": ["class Solution:\n    def answer(self):\n        return 42"],
        }
    )

    assert repaired["code_list"] == ["class Solution:\n    def answer(self):\n        return 42"]
    assert reports[0]["has_entrypoint"] is True


def test_candidate_health_audit_classifies_public_rejected_valid_candidates(
    tmp_path: Path,
) -> None:
    audit = load_candidate_health_audit_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "generations.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_title": "task",
                    "difficulty": "medium",
                    "code_list": [
                        "class Solution:\n    def answer(self):\n        return 1",
                        "class Solution:\n    def answer(self):\n        return 2",
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "public_selection.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "selected_index": 0,
                        "public_scores": [0.0, 0.0],
                        "public_oracle_pass": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "eval_all.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "graded_list": [False],
                    "metadata": [{"error_code": -2, "error_message": "Wrong Answer"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = audit.summarize_run(run_dir)

    assert summary["totals"]["syntax_ok_candidates"] == 2
    assert summary["records"][0]["failure_mode"] == "public_tests_reject_syntax_valid_candidates"
    assert summary["records"][0]["selected_error_code"] == -2


def test_candidate_health_audit_classifies_no_syntax_valid_candidates(
    tmp_path: Path,
) -> None:
    audit = load_candidate_health_audit_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "generations.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "code_list": ["class Solution:\n    def answer(self,"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "public_selection.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "selected_index": 0,
                        "public_scores": [0.0],
                        "public_oracle_pass": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = audit.summarize_run(run_dir)

    assert summary["totals"]["syntax_ok_candidates"] == 0
    assert summary["records"][0]["failure_mode"] == "no_syntax_valid_candidates"


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


def test_evaluator_conservative_behavior_policy_requires_public_pass() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_conservative_behavior_selected_index_from_scores(
        public_scores=[1.0, 0.0],
        code_outputs=["short public pass", "better behavior"],
        behavior_outputs=[
            ["ERR:1:aaa", "ERR:1:bbb"],
            ["OK:1:aaa", "OK:1:bbb"],
        ],
        tie_breaker="shortest",
        min_behavior_tests=2,
        min_behavior_success_rate=1.0,
        min_behavior_consensus_margin=0,
        min_behavior_cluster_margin=0,
    )

    assert selected == 0


def test_evaluator_conservative_behavior_policy_allows_strong_public_pass_override() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_conservative_behavior_selected_index_from_scores(
        public_scores=[1.0, 1.0],
        code_outputs=["short", "longer behavior candidate"],
        behavior_outputs=[
            ["ERR:1:aaa", "ERR:1:bbb"],
            ["OK:1:aaa", "OK:1:bbb"],
        ],
        tie_breaker="shortest",
        min_behavior_tests=2,
        min_behavior_success_rate=1.0,
        min_behavior_consensus_margin=0,
        min_behavior_cluster_margin=0,
    )

    assert selected == 1


def test_evaluator_conservative_behavior_policy_rejects_weak_behavior_signal() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_conservative_behavior_selected_index_from_scores(
        public_scores=[1.0, 1.0],
        code_outputs=["short", "longer behavior candidate"],
        behavior_outputs=[
            ["ERR:1:aaa", "ERR:1:bbb"],
            ["OK:1:aaa", "ERR:1:bbb"],
        ],
        tie_breaker="shortest",
        min_behavior_tests=2,
        min_behavior_success_rate=1.0,
        min_behavior_consensus_margin=0,
        min_behavior_cluster_margin=0,
    )

    assert selected == 0


def test_evaluator_differential_medoid_uses_public_pass_clusters() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_differential_medoid_selected_index_from_scores(
        public_scores=[1.0, 1.0, 1.0, 0.0],
        code_outputs=["long public code", "ok", "other cluster", "bad"],
        behavior_outputs=[
            ["OK:1:aaa", "OK:1:bbb"],
            ["OK:1:aaa", "OK:1:bbb"],
            ["OK:1:ccc", "OK:1:ddd"],
            ["OK:1:aaa", "OK:1:bbb"],
        ],
        tie_breaker="shortest",
    )

    assert selected == 1


def test_evaluator_differential_medoid_handles_missing_behavior_outputs() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_differential_medoid_selected_index_from_scores(
        public_scores=[1.0, 1.0, 1.0],
        code_outputs=["public", "candidate", "missing behavior result"],
        behavior_outputs=[
            ["OK:1:aaa"],
            ["OK:1:bbb"],
        ],
        tie_breaker="shortest",
    )

    assert selected == 0


def test_evaluator_conservative_differential_medoid_requires_cluster_margin() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_conservative_differential_medoid_selected_index_from_scores(
        public_scores=[1.0, 1.0, 1.0],
        code_outputs=["public", "cluster peer", "other"],
        behavior_outputs=[
            ["OK:1:aaa", "OK:1:bbb"],
            ["OK:1:aaa", "OK:1:bbb"],
            ["OK:1:ccc", "OK:1:ddd"],
        ],
        tie_breaker="shortest",
        min_behavior_tests=2,
        min_behavior_success_rate=1.0,
        min_behavior_cluster_margin=1,
        min_differential_tests=2,
    )

    assert selected == 0


def test_evaluator_conservative_differential_medoid_allows_strong_override() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_conservative_differential_medoid_selected_index_from_scores(
        public_scores=[1.0, 1.0, 1.0, 1.0],
        code_outputs=["public", "cluster peer a", "x", "cluster peer b"],
        behavior_outputs=[
            ["OK:1:aaa", "OK:1:bbb"],
            ["OK:1:ccc", "OK:1:ddd"],
            ["OK:1:xxx", "OK:1:yyy"],
            ["OK:1:ccc", "OK:1:ddd"],
        ],
        tie_breaker="shortest",
        min_behavior_tests=2,
        min_behavior_success_rate=1.0,
        min_behavior_cluster_margin=1,
        min_differential_tests=2,
    )

    assert selected == 1


def test_evaluator_differential_support_uses_public_failed_cluster_votes() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_differential_support_selected_index_from_scores(
        public_scores=[1.0, 1.0, 0.0, 0.0],
        code_outputs=["public", "supported", "failed peer a", "failed peer b"],
        behavior_outputs=[
            ["OK:1:aaa", "OK:1:bbb"],
            ["OK:1:ccc", "OK:1:ddd"],
            ["OK:1:ccc", "OK:1:ddd"],
            ["OK:1:ccc", "OK:1:ddd"],
        ],
        tie_breaker="shortest",
    )

    assert selected == 1


def test_evaluator_conservative_differential_support_requires_margin() -> None:
    evaluator = load_evaluator_module()

    selected = evaluator.choose_conservative_differential_support_selected_index_from_scores(
        public_scores=[1.0, 1.0, 0.0],
        code_outputs=["public", "supported", "failed peer"],
        behavior_outputs=[
            ["OK:1:aaa", "OK:1:bbb"],
            ["OK:1:ccc", "OK:1:ddd"],
            ["OK:1:ccc", "OK:1:ddd"],
        ],
        tie_breaker="shortest",
        min_behavior_tests=2,
        min_behavior_success_rate=1.0,
        min_behavior_cluster_margin=2,
        min_differential_tests=2,
    )

    assert selected == 0


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


def test_adaptive_differential_stdin_mutations_are_deterministic() -> None:
    adaptive = load_adaptive_differential_module()

    first = adaptive.adaptive_stdin_mutations(
        "3\n1 2 3\n",
        rng=adaptive.random.Random(7),
        limit=12,
        max_input_chars=200,
    )
    second = adaptive.adaptive_stdin_mutations(
        "3\n1 2 3\n",
        rng=adaptive.random.Random(7),
        limit=12,
        max_input_chars=200,
    )

    assert first == second
    assert len(first) == 12
    assert "3\n1 2 3\n" not in first


def test_adaptive_differential_chooses_only_public_pass_differences() -> None:
    adaptive = load_adaptive_differential_module()

    inputs, indices = adaptive.choose_differential_inputs(
        behavior_inputs=["a", "b", "c"],
        behavior_outputs=[
            ["OK:1:x", "OK:1:y", "ERR:1:z"],
            ["OK:1:x", "OK:1:z", "OK:1:k"],
            ["OK:1:q", "OK:1:z", "OK:1:k"],
        ],
        candidate_indices=[0, 1],
        max_output_inputs=8,
    )

    assert inputs == ["b"]
    assert indices == [1]


def test_expected_verifier_builds_deduplicated_output_options() -> None:
    verifier = load_expected_verifier_module()

    options = verifier.build_output_options(
        [
            {"status": "OK", "value": "4\n"},
            {"status": "OK", "value": "5\n"},
            {"status": "OK", "value": "4\n"},
            {"status": "ERR", "error": "timeout"},
        ],
        max_option_chars=100,
    )

    assert [option["label"] for option in options] == ["A", "B"]
    assert options[0]["candidate_indices"] == [0, 2]
    assert options[1]["candidate_indices"] == [1]


def test_expected_verifier_parses_choice_payloads() -> None:
    verifier = load_expected_verifier_module()

    parsed = verifier.parse_choice_payload(
        '```json\n{"choice": "option b", "confidence": 1.7, "reason": "x"}\n```'
    )

    assert parsed == {"choice": "B", "confidence": 1.0, "reason": "x"}


def test_expected_verifier_selector_requires_confidence_margin() -> None:
    selector = load_expected_selector_module()
    public_payload = {
        "records": [
            {
                "question_id": "task",
                "selected_index": 0,
                "public_scores": [1.0, 1.0, 0.0],
                "selected_public_score": 1.0,
            }
        ]
    }
    candidate_payload = {
        "candidate_outputs": [
            {
                "question_id": "task",
                "input_index": 0,
                "options": [
                    {"label": "A", "candidate_indices": [0]},
                    {"label": "B", "candidate_indices": [1, 2]},
                ],
            },
            {
                "question_id": "task",
                "input_index": 1,
                "options": [
                    {"label": "A", "candidate_indices": [0]},
                    {"label": "B", "candidate_indices": [1]},
                ],
            },
        ]
    }
    choices = {
        "records": [
            {
                "record_id": selector.prompt_id("task", 0),
                "question_id": "task",
                "input_index": 0,
                "choice": "B",
                "confidence": 0.7,
            },
            {
                "record_id": selector.prompt_id("task", 1),
                "question_id": "task",
                "input_index": 1,
                "choice": "B",
                "confidence": 0.8,
            },
        ]
    }

    selected = selector.build_expected_verifier_selection(
        public_payload,
        candidate_payload,
        choices,
        min_choice_count=2,
        min_confidence_margin=1.0,
    )

    record = selected["records"][0]
    assert record["selected_index"] == 1
    assert record["override_from_public"] is True
    assert record["verifier_confidence_margin_vs_public"] == 1.5


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


def test_behavior_generator_parses_llm_outputs() -> None:
    generator = load_behavior_generator_module()

    records = generator.parse_behavior_inputs_from_outputs(
        [
            {
                "question_id": "a",
                "raw_output": '```json\n{"inputs": ["1\\n2", "", "3"]}\n```',
            },
            {"question_id": "b", "raw_output": "not json"},
        ],
        max_inputs=2,
        max_input_chars=20,
    )

    assert records == [
        {
            "question_id": "a",
            "inputs": ["1\n2", "3"],
            "n_inputs": 2,
            "source": "local_model_candidate_aware_differential_v1",
        },
        {
            "question_id": "b",
            "inputs": [],
            "n_inputs": 0,
            "source": "local_model_candidate_aware_differential_v1",
        },
    ]


def test_behavior_generator_repairs_invalid_json_escapes() -> None:
    generator = load_behavior_generator_module()

    records = generator.parse_behavior_inputs_from_outputs(
        [
            {
                "question_id": "a",
                "raw_output": '```json\n{"inputs": ["6\\nabc\\acb\\n"]}\n```',
            }
        ],
        max_inputs=4,
        max_input_chars=100,
    )

    assert records[0]["n_inputs"] == 1
    assert records[0]["inputs"] == ["6\nabc\\acb"]
