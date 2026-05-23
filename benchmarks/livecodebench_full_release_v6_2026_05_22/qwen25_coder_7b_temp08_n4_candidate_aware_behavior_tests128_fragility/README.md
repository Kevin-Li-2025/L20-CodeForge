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

This batch has generated input-only behavior tests. It has not yet been used as
a headline LiveCodeBench score. The next measurement step is to run the
conservative behavior selector against the full candidate pool and then perform
a retry-stabilized hidden-test replay.
