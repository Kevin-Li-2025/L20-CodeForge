# Expected-Output Verifier Prompts

This directory records the first expected-output verifier stage built from the
adaptive differential fuzz inputs.

## Purpose

Input-only clustering improved differential coverage but did not identify which
side of a disagreement was correct. This verifier stage converts each
differential input into a multiple-choice expected-output prompt:

1. Execute all `n=4` candidates on the input.
2. Deduplicate successful candidate outputs into labeled options.
3. Ask a verifier model to choose the correct option or `NONE`.
4. Convert verifier choices into conservative candidate-selection overrides.

No hidden expected outputs are used to build these prompts.

## Artifact Summary

- Differential inputs consumed: `154`.
- Prompt records: `154`.
- Tasks covered: `34`.
- Output-option distribution: `97` prompts with 2 options, `50` with 3 options,
  and `7` with 4 options.
- Candidate output extraction wall time: `57.393s`.

Primary files:

- `prompts.jsonl`: verifier prompt records keyed by `record_id`.
- `candidate_outputs.json`: raw candidate outputs and deduplicated options.
- `dry_run/manifest.json`: local dry-run manifest for the L20 verifier generator.
- `l20_qwen25_coder_7b_choices/`: completed L20 verifier choices, selection,
  hidden replay comparison, and override audit.

## L20 Verifier Result

The L20 Qwen2.5-Coder-7B verifier generated and parsed all `154/154` prompt
records in `292.974s`.

Choice distribution:

| Choice | Count |
| --- | ---: |
| `B` | 63 |
| `NONE` | 43 |
| `A` | 37 |
| `C` | 10 |
| `D` | 1 |

The conservative selector used `min_choice_count=2` and
`min_confidence_margin=1.0`. It made `10` public-pass overrides on the
112-target replay.

| Run | Passed | Total | Status |
| --- | ---: | ---: | --- |
| Same target IDs, public-selection baseline | 68 | 112 | baseline |
| Expected-output verifier selector, raw replay | 65 | 112 | regressed |
| Expected-output verifier selector, rechecked audit | 67 | 112 | regressed |

Override audit:

- Improvements: `1` (`2854`, selected index `3 -> 1`).
- Regressions: `2` (`abc355_a`, selected index `1 -> 0`; `abc366_a`, selected
  index `0 -> 3`).
- Neutral overrides: `7`.
- Unchanged-code flaky tasks in the raw replay: `abc363_c`, `abc378_e`; both
  stabilize under the existing recheck payload.

The threshold sweep in
`l20_qwen25_coder_7b_choices/expected_verifier_override_audit.json` shows that
simple count/margin tightening does not solve the problem. The two regressions
had high verifier confidence margins (`8.0` and `5.0`), so this verifier is
overconfident on some expected-output judgments. The artifact is useful as a
calibration set, but it should not be merged into the headline `378/1055`
LiveCodeBench result.

## Reproduction Commands

Run on the L20 host after pulling the repo:

```bash
python scripts/generate_lcb_verifier_choices.py \
  --prompts benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_expected_output_verifier_prompts154/prompts.jsonl \
  --output-dir artifacts/lcb_expected_verifier/qwen25_coder_7b_prompts154 \
  --model /path/to/Qwen2.5-Coder-7B-Instruct \
  --resume
```

Then convert choices to a conservative selection payload:

```bash
python scripts/select_lcb_expected_verifier_candidates.py \
  --public-selection benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_public_select_full_eval/public_selection.json \
  --candidate-outputs benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_expected_output_verifier_prompts154/candidate_outputs.json \
  --verifier-choices artifacts/lcb_expected_verifier/qwen25_coder_7b_prompts154/verifier_choices.json \
  --output artifacts/lcb_expected_verifier/qwen25_coder_7b_prompts154/expected_verifier_selection.json
```

The resulting selection payload can be replayed through
`scripts/evaluate_lcb_generations.py` with `--public-selection` for hidden-test
measurement.

The completed audit can be regenerated with:

```bash
python scripts/audit_lcb_expected_verifier.py \
  --comparison-summary benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_expected_output_verifier_prompts154/l20_qwen25_coder_7b_choices/expected_verifier_targets112_rechecked_summary.json \
  --public-selection benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_public_select_full_eval/public_selection.json \
  --candidate-outputs benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_expected_output_verifier_prompts154/candidate_outputs.json \
  --verifier-choices benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_expected_output_verifier_prompts154/l20_qwen25_coder_7b_choices/verifier_choices.json \
  --candidate-selection benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_expected_output_verifier_prompts154/l20_qwen25_coder_7b_choices/expected_verifier_selection.json \
  --baseline-eval-all benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_public_select_full_eval/eval_all.json \
  --candidate-eval-all benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_expected_output_verifier_targets112_eval/eval_all.json \
  --output benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_expected_output_verifier_prompts154/l20_qwen25_coder_7b_choices/expected_verifier_override_audit.json
```
