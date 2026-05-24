from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


def load_runner_module():
    script = Path(__file__).parents[1] / "scripts" / "run_lcb_subset_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_lcb_subset_benchmark", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeProblem:
    question_content: str
    starter_code: str = ""


def test_build_lcb_generation_prompt_matches_stdin_format() -> None:
    runner = load_runner_module()

    prompt = runner.build_lcb_generation_prompt(FakeProblem("Solve A+B."))

    assert "### Question:" in prompt
    assert "Solve A+B." in prompt
    assert "Read the inputs from stdin" in prompt
    assert "# YOUR CODE HERE" in prompt
    assert "### Answer:" in prompt


def test_build_lcb_generation_prompt_includes_starter_code() -> None:
    runner = load_runner_module()

    prompt = runner.build_lcb_generation_prompt(
        FakeProblem("Implement add.", "class Solution:\n    pass")
    )

    assert "starter code" in prompt
    assert "class Solution" in prompt
    assert "# YOUR CODE HERE" not in prompt


def test_build_lcb_generation_prompt_appends_prompt_suffix() -> None:
    runner = load_runner_module()

    prompt = runner.build_lcb_generation_prompt(
        FakeProblem("Solve A+B."),
        prompt_suffix="Return only final code.",
    )

    assert prompt.endswith("Return only final code.\n\n")


def test_resolve_response_prefix_literal_mode_preserves_text() -> None:
    runner = load_runner_module()

    prefix = runner.resolve_response_prefix(
        FakeProblem("Solve A+B.", "class Solution:\n    def add(self):"),
        response_prefix="```python\n",
        response_prefix_mode="literal",
    )

    assert prefix == "```python\n"


def test_resolve_response_prefix_starter_code_mode_appends_signature() -> None:
    runner = load_runner_module()

    prefix = runner.resolve_response_prefix(
        FakeProblem("Solve A+B.", "class Solution:\n    def add(self, a: int) -> int:"),
        response_prefix="```python\nfrom typing import List",
        response_prefix_mode="starter-code",
    )

    assert prefix == (
        "```python\n"
        "from typing import List\n"
        "class Solution:\n"
        "    def add(self, a: int) -> int:\n"
    )


def test_resolve_response_prefix_starter_code_mode_uses_code_fence_default() -> None:
    runner = load_runner_module()

    prefix = runner.resolve_response_prefix(
        FakeProblem("Solve A+B."),
        response_prefix="",
        response_prefix_mode="starter-code",
    )

    assert prefix == "```python\n"


def test_strip_lcb_code_block_prefers_last_python_like_block() -> None:
    runner = load_runner_module()

    output = (
        "Idea\n```text\nnot code\n```\n"
        "```python\nimport sys\nprint(sys.stdin.read())\n```\n"
    )

    assert runner.strip_lcb_code_block(output) == "import sys\nprint(sys.stdin.read())"


def test_generated_text_has_closed_code_block_requires_two_fences() -> None:
    runner = load_runner_module()

    assert runner.generated_text_has_closed_code_block("```python\nprint(1)\n```") is True
    assert runner.generated_text_has_closed_code_block("```python\nprint(1)") is False


def test_sanitize_lcb_metadata_removes_test_payloads() -> None:
    runner = load_runner_module()
    payload = {
        "in" + "puts": "hidden",
        "ex" + "pected": "42",
        "error_code": -2,
        "error_message": "Wrong",
    }

    sanitized = runner.sanitize_lcb_metadata([json.dumps(payload)])

    assert sanitized == [{"error_code": -2, "error_message": "Wrong"}]


def test_choose_public_selected_index_prefers_short_public_pass() -> None:
    runner = load_runner_module()

    selected = runner.choose_public_selected_index(
        public_results=[[False, True], [True, True], [True, True]],
        code_outputs=["bad", "longer passing code", "ok"],
        tie_breaker="shortest",
    )

    assert selected == 2


def test_choose_public_selected_index_uses_best_partial_score() -> None:
    runner = load_runner_module()

    selected = runner.choose_public_selected_index(
        public_results=[[True, False, False], [True, True, False], [False, False, False]],
        code_outputs=["a", "bb", "c"],
        tie_breaker="first",
    )

    assert selected == 1


def test_public_tie_breaker_prefers_syntax_valid_candidate() -> None:
    runner = load_runner_module()

    selected = runner.choose_public_selected_index(
        public_results=[[False], [False]],
        code_outputs=[
            "class Solution:\n    def answer(self,",
            "class Solution:\n    def answer(self):\n        return 1",
        ],
        tie_breaker="shortest",
    )

    assert selected == 1


def test_count_static_healthy_candidates_requires_syntax_and_entrypoint() -> None:
    runner = load_runner_module()

    assert (
        runner.count_static_healthy_candidates(
            [
                "class Solution:\n    def answer(self):\n        return 1",
                "class Solution:\n    def answer(self,",
                "def helper():\n    return 1",
            ]
        )
        == 1
    )


def test_pad_generations_for_lcb_metrics_rectangularizes_variable_counts() -> None:
    runner = load_runner_module()

    padded, did_pad = runner.pad_generations_for_lcb_metrics([["a", "b"], ["c"]])

    assert did_pad is True
    assert padded == [["a", "b"], ["c", "c"]]


def test_trim_metric_payloads_to_candidate_counts_removes_padding() -> None:
    runner = load_runner_module()

    results = runner.trim_metric_results_to_candidate_counts(
        {0: [["r0"], ["r1"]], 1: [["r2"], ["padded"]]},
        [2, 1],
    )
    metadata = runner.trim_metric_metadata_to_candidate_counts(
        [["m0", "m1"], ["m2", "padded"]],
        [2, 1],
    )

    assert results == {0: [["r0"], ["r1"]], 1: [["r2"]]}
    assert metadata == [["m0", "m1"], ["m2"]]


def test_generate_problem_outputs_static_retry_adds_healthy_candidate() -> None:
    runner = load_runner_module()
    outputs = [
        "class Solution:\n    def answer(self,",
        "class Solution:\n    def answer(self):\n        return 1",
    ]
    progress_counts = []

    def fake_generate_one_batch(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return [outputs.pop(0)]

    runner.generate_one_batch = fake_generate_one_batch

    raw_outputs, code_outputs = runner.generate_problem_outputs(
        model=object(),
        tokenizer=object(),
        problem=FakeProblem("Implement answer."),
        n_samples=1,
        sample_batch_size=1,
        prompt_suffix="",
        prompt_rendering="raw",
        system_message="",
        response_prefix="",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=32,
        max_input_tokens=128,
        stop_after_code_block=False,
        min_static_healthy_samples=1,
        max_extra_static_retry_samples=2,
        existing_code_outputs_for_health=[],
        progress_callback=lambda raw, code: progress_counts.append((len(raw), len(code))),
    )

    assert raw_outputs == [
        "class Solution:\n    def answer(self,",
        "class Solution:\n    def answer(self):\n        return 1",
    ]
    assert code_outputs == raw_outputs
    assert progress_counts == [(1, 1), (2, 2)]


def test_build_public_selection_records_returns_single_selected_generation() -> None:
    runner = load_runner_module()
    problem = type("Problem", (), {"question_id": "x", "question_title": "title"})()

    selected, records = runner.build_public_selection_records(
        problems=[problem],
        generations=[["bad", "good"]],
        public_results={0: [[False], [True]]},
        tie_breaker="first",
    )

    assert selected == [["good"]]
    assert records[0]["selected_index"] == 1
    assert records[0]["public_oracle_pass"] is True


def test_build_base_model_kwargs_places_non_quantized_model_on_cuda() -> None:
    runner = load_runner_module()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        bfloat16 = "bf16"
        float16 = "fp16"
        cuda = FakeCuda()

    kwargs = runner.build_base_model_kwargs(
        torch_module=FakeTorch,
        load_in_4bit=False,
        bf16=True,
        attn_implementation="sdpa",
    )

    assert kwargs["torch_dtype"] == "bf16"
    assert kwargs["device_map"] == "auto"
    assert kwargs["attn_implementation"] == "sdpa"


def test_parse_question_ids_strips_empty_values() -> None:
    runner = load_runner_module()

    assert runner.parse_question_ids(" 2777, ,2784 ") == {"2777", "2784"}


def test_build_generation_record_includes_partial_outputs() -> None:
    runner = load_runner_module()
    problem = SimpleNamespace(
        question_id="2784",
        question_title="power",
        contest_date=SimpleNamespace(isoformat=lambda: "2023-05-13T00:00:00"),
        platform=SimpleNamespace(value="leetcode"),
        difficulty=SimpleNamespace(value="hard"),
        question_content="Solve it.",
        starter_code="",
    )

    record = runner.build_generation_record(
        problem=problem,
        prompt_suffix="Return code only.",
        response_prefix="```python\n",
        response_prefix_mode="literal",
        raw_outputs=["raw"],
        code_outputs=["code"],
    )

    assert record["question_id"] == "2784"
    assert record["raw_outputs"] == ["raw"]
    assert record["code_list"] == ["code"]
    assert record["response_prefix"] == "```python\n"
    assert record["response_prefix_mode"] == "literal"
    assert record["prompt"].endswith("Return code only.\n\n")


def test_validate_resume_records_truncates_to_requested_samples() -> None:
    runner = load_runner_module()

    records = [
        {
            "question_id": "abc",
            "raw_outputs": ["a", "b", "c"],
            "code_list": ["A", "B", "C"],
        }
    ]

    resumed = runner.validate_resume_records(records, n_samples=2)

    assert resumed["abc"]["raw_outputs"] == ["a", "b"]
    assert resumed["abc"]["code_list"] == ["A", "B"]


def test_validate_resume_records_rejects_partial_samples() -> None:
    runner = load_runner_module()

    records = [{"question_id": "abc", "raw_outputs": ["a"], "code_list": ["A"]}]

    try:
        runner.validate_resume_records(records, n_samples=2)
    except ValueError as exc:
        assert "fewer than 2 samples" in str(exc)
    else:
        raise AssertionError("expected ValueError for partial resume record")


def test_validate_resume_records_allows_partial_extension() -> None:
    runner = load_runner_module()

    records = [{"question_id": "abc", "raw_outputs": ["a"], "code_list": ["A"]}]

    resumed = runner.validate_resume_records(
        records,
        n_samples=2,
        allow_partial=True,
    )

    assert resumed["abc"]["raw_outputs"] == ["a"]
    assert resumed["abc"]["code_list"] == ["A"]


def test_load_resume_generation_records_prefers_output_file(tmp_path: Path) -> None:
    runner = load_runner_module()
    output = tmp_path / "output_generations.json"
    source = tmp_path / "source_generations.json"
    output.write_text('[{"question_id": "same", "code_list": ["new"]}]', encoding="utf-8")
    source.write_text('[{"question_id": "same", "code_list": ["old"]}]', encoding="utf-8")

    records, source_path = runner.load_resume_generation_records(output, source)

    assert records == [{"question_id": "same", "code_list": ["new"]}]
    assert source_path == f"{output} over {source}"


def test_load_resume_generation_records_uses_seed_when_output_missing(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    output = tmp_path / "output_generations.json"
    source = tmp_path / "source_generations.json"
    source.write_text('[{"question_id": "source"}]', encoding="utf-8")

    records, source_path = runner.load_resume_generation_records(output, source)

    assert records == [{"question_id": "source"}]
    assert source_path == str(source)


def test_load_resume_generation_records_merges_partial_output_with_seed(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    output = tmp_path / "output_generations.json"
    source = tmp_path / "source_generations.json"
    source.write_text(
        '[{"question_id": "a", "code_list": ["old-a"]},'
        ' {"question_id": "b", "code_list": ["old-b"]}]',
        encoding="utf-8",
    )
    output.write_text(
        '[{"question_id": "a", "code_list": ["new-a"]}]',
        encoding="utf-8",
    )

    records, _ = runner.load_resume_generation_records(output, source)

    assert records == [
        {"question_id": "a", "code_list": ["new-a"]},
        {"question_id": "b", "code_list": ["old-b"]},
    ]
