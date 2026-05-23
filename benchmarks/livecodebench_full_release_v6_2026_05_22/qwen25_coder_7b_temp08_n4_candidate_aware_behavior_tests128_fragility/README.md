# Public-Fragility Behavior-Test Batch

This directory records the second generated behavior-test batch for the full
LiveCodeBench `release_v6` `n=4` candidate pool.

## Targeting

- Target priority: `public-fragility`.
- Prompt records: `128`.
- Target source: public-test scores only.
- Hidden-test labels were not used to build prompts or select targets.

The targeter prioritizes public-passing ties where fewer candidates pass all
public tests and more candidates partially pass public tests. A retrospective
audit in `target_priority_analysis_2026_05_23` showed that this raises the
64-prompt public-pass/hidden-fail target density from `3/64` under input order
to `20/64`.

## Generation

- Model: `Qwen2.5-Coder-7B-Instruct`.
- Prompt count: `128`.
- Generated records: `128`.
- Parsed non-empty behavior records: `112`.
- Parsed behavior inputs: `1130`.
- Wall time: `1227.057s`.

## Status

This batch has generated input-only behavior tests and has now been measured on
the `112` parsed target tasks with the conservative public-pass behavior
selector.

Targeted hidden-test result:

| Run | Passed | Total | Status |
| --- | ---: | ---: | --- |
| Public-selection baseline on same target IDs | 68 | 112 | baseline |
| Conservative behavior selector, raw replay | 66 | 112 | unstable replay |
| Conservative behavior selector, rechecked audit | 68 | 112 | stabilized neutral |

The selector made `4` public-pass overrides. None changed a hidden outcome. The
raw `-2` came from the same two unchanged-code flaky tasks already seen in the
prior full hybrid audit (`abc363_c`, `abc378_e`); the existing recheck payload
stabilizes both back to pass.

This is therefore not a new headline LiveCodeBench score. The useful result is
negative: generated input-only consensus is not yet strong enough on this
high-risk batch to beat public selection. The next measurement should either
tighten the override margin or add an expected-output/verifier stage before a
full merged replay.
