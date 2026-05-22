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

This is not a leaderboard score. The subset is deliberately small so a single
L20 can run frequent checkpoints. The value is the strict hidden-test signal:
we can now measure whether future public-test selection, repair, and RL/SFT
changes improve or regress on real LCB tasks.

## Immediate Diagnosis

The 3-problem smoke run found two real model failures:

- `1873_A`: the model overcomplicated a simple swap rule and failed sample/hidden
  behavior.
- `2754`: the model mishandled negative-product parity.

The 20-problem run recovered to `0.600` pass@1, which is useful but leaves a
large gap for inference-time methods. The next high-leverage experiment is an
LCB public-test selector: sample several candidates per problem, run only the
provided public tests for selection, then grade the chosen candidate with the
official hidden-test evaluator.

## Next Experiments

1. Add a transparent `--public-select` mode to the LCB runner.
2. Compare greedy `n=1` against temperature-sampled `n=4` and `n=8` with public
   selection.
3. Add lightweight repair prompts for candidates that fail public tests.
4. Record pass@1, public-pass rate, hidden-pass rate, wall time, and tokens so
   improvements are attributable to algorithmic test-time compute, not hidden
   leakage.
