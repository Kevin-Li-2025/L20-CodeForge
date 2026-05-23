# Candidate-Aware Behavior Tests 64

This directory records the first model-generated behavior-test pass for the
full LiveCodeBench `release_v6` `n=4` public-selection run.

## Generation

- Prompt bank:
  `qwen25_coder_7b_temp08_n4_candidate_aware_behavior_prompts64/prompts.jsonl`.
- Test generator: `Qwen2.5-Coder-7B-Instruct`, local 4-bit Transformers
  inference on the L20.
- Prompt records: `64`.
- Raw model outputs: `64`.
- Parsed behavior-test records: `54`.
- Generated behavior inputs: `558`.
- Generation wall time: `580.994s`.

The generated tests are input-only. They do not contain hidden-test outputs.

## Targeted Result

The 54 tasks with parsed behavior inputs were evaluated as a targeted selector
probe.

| Selector | Passed | Total | pass@1 |
| --- | ---: | ---: | ---: |
| Reused public-test selection | 51 | 54 | 0.9444 |
| Public score + generated behavior consensus | 52 | 54 | 0.9630 |
| Conservative public-pass behavior override, margin 1 | 52 | 54 | 0.9630 |

Selection changes:

- `2837`: selected index changed `1 -> 3`; hidden result stayed pass.
- `2848`: selected index changed `3 -> 1`; hidden result stayed pass.
- `2854`: selected index changed `3 -> 1`; hidden result changed fail to pass.

The conservative override policy requires:

- The replacement candidate must also pass public tests.
- At least `6` behavior tests must be available.
- Behavior success rate must be at least `0.9`.
- Behavior consensus must beat the public-selected candidate by at least `1`.

The stricter policy still made the same three overrides and preserved the
`+1/54` targeted gain. The audit summary is
`conservative_m1_delta_summary.json`.

This is a real positive signal, but it is still a small targeted probe. It
should not be reported as a new full-suite score by itself.

## Full Hybrid Check

A full-suite hybrid selection was also evaluated by replacing the 54 target
records in the existing public-selection payload. That single full run produced
`377/1055`, but the diff showed two unrelated non-target tasks with unchanged
selected indices regressed during the full execution:

- `abc363_c`
- `abc378_e`

Both tasks passed when immediately rechecked as a two-task subset, so the full
hybrid run is treated as an execution-stability warning rather than a valid
model regression. The main full-suite score remains the prior stable
public-selection result, `378/1055`, until a retry-stabilized full evaluator is
in place. The automated comparison summary is
`full_hybrid_stability_summary.json`, with status `unstable_replay`.

## Next Gate

Before adopting generated behavior tests into the main score:

1. Add retry or majority recheck for candidates whose selected code is unchanged
   but hidden outcome changes across runs.
2. Apply the conservative override policy to full-suite replay.
3. Re-run full hybrid evaluation with the stabilized evaluator.
