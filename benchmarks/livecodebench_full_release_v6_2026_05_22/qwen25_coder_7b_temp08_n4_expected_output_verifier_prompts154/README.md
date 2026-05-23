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

## Next Command

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
