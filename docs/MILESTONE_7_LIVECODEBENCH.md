# Milestone 7: LiveCodeBench Strict Subset

The project now has a second public benchmark line beyond EvalPlus. The new
runner uses local HF inference on the L20 and delegates grading to the official
LiveCodeBench `codegen_metrics` evaluator.

## Why This Matters

EvalPlus is useful for function-level robustness, but it is not enough to show
general contest-style coding ability. LiveCodeBench adds newer problems,
stdin/stdout tasks, hidden tests, and LeetCode-style functional tasks. This is a
better proxy for whether post-training and test-time compute methods transfer
outside HumanEval/MBPP.

## Current Checkpoint

- Benchmark: `livecodebench/code_generation_lite`, `release_v6`.
- Scope: fixed 20-problem compact-test subset from shard
  `test-00000-of-00009.parquet`.
- Official evaluator repo commit:
  `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.
- Model: `Qwen2.5-Coder-7B-Instruct`, 4-bit NF4, bf16 compute, greedy decoding.
- Result: `12/20`, pass@1 `0.600`.
- Runtime: `106.201s` generation, `8.606s` evaluation.

The first test-time compute run uses temperature `0.8`, `n=4`, and public-test
selection:

- Result: `17/20`, pass@1 `0.850`.
- Runtime: `227.834s` generation, `2.230s` public selection, `8.562s` final
  evaluation.
- Public-test oracle: `19/20`.
- Residual failures: `2754`, `2837`, `3017`.

Increasing the same setup to `n=8`:

- Result: `18/20`, pass@1 `0.900`.
- Runtime: `332.421s` generation, `5.271s` public selection, `8.558s` final
  evaluation.
- Public-test oracle: `20/20`.
- Residual failures: `2754`, `3017`.

This is not a leaderboard score. The subset is deliberately small so a single
L20 can run frequent checkpoints. The value is the strict hidden-test signal:
we can now measure whether future public-test selection, repair, and RL/SFT
changes improve or regress on real LCB tasks.

## Immediate Diagnosis

The 3-problem smoke run found two real model failures:

- `1873_A`: the model overcomplicated a simple swap rule and failed sample/hidden
  behavior.
- `2754`: the model mishandled negative-product parity.

The 20-problem run recovered to `0.600` pass@1, and public-test selection lifted
that to `0.850` with `n=4` and `0.900` with `n=8`. This is a strong algorithmic
signal: on this fixed subset, small-sample test-time compute is worth `+30`
points without changing model weights. The `n=8` public-test oracle of `20/20`
also shows that more gains are likely available from better hidden-robust
selection or repair, while larger sampling alone is starting to show diminishing
returns.

## Next Experiments

1. Add lightweight repair prompts for candidates that fail public tests.
2. Add a second selector signal, such as shortest-public-pass plus diversity or
   multi-candidate consensus on public tests.
3. Scale the same `n=4` and `n=8` protocol to a larger 100-problem LCB subset.
4. Record pass@1, public-pass rate, hidden-pass rate, wall time, and tokens so
   improvements are attributable to algorithmic test-time compute, not hidden
   leakage.
