# Milestone 7: LiveCodeBench Strict Subset and Full Run

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

## Full `release_v6` Checkpoint

A full-suite greedy baseline is now recorded under
`benchmarks/livecodebench_full_release_v6_2026_05_22/`.

- Scope: all 1,055 `release_v6` tasks.
- Evaluation: saved generations graded locally with official LiveCodeBench
  hidden/private tests.
- Model: `Qwen2.5-Coder-7B-Instruct`, 4-bit NF4, bf16 compute, no adapter.
- Decode: greedy, `n=1`, `temperature=0`, `max_new_tokens=2048`.
- Result: `297/1055`, pass@1 `0.2815`.
- Generation runtime: `6795.360s`.
- Evaluation runtime: `467.664s`.
- Full JSONL SHA-256:
  `4519422b52d7dd243358ee721ecfe94e26ab364a7af2938c8e5d170a17bcadcd`.

Breakdown:

- Easy: `206/322`, pass@1 `0.6398`.
- Medium: `82/383`, pass@1 `0.2141`.
- Hard: `9/350`, pass@1 `0.0257`.
- AtCoder: `146/602`, pass@1 `0.2425`.
- LeetCode: `149/444`, pass@1 `0.3356`.

This is a stricter and more reliable baseline than the 20-task subset. The
subset result was useful for fast iteration, but it was optimistic; the full
run shows that medium/hard contest reasoning is the real bottleneck.

## Immediate Diagnosis

The 3-problem smoke run found two real model failures:

- `1873_A`: the model overcomplicated a simple swap rule and failed sample/hidden
  behavior.
- `2754`: the model mishandled negative-product parity.

The 20-problem run recovered to `0.600` pass@1, and public-test selection lifted
that to `0.850` with `n=4` and `0.900` with `n=8`. This is a strong algorithmic
signal on the subset, but the full greedy score of `0.2815` makes the caveat
clear: the small subset cannot be treated as a public benchmark score.

On the full run, failures are mostly wrong answers:

- Wrong answer: `605`.
- Runtime error: `85`.
- Time limit exceeded: `68`.

That points to repair, verifier-guided selection, and curriculum targeted at
medium/hard algorithmic reasoning before spending GPU time on broader SFT.

## Next Experiments

1. Add lightweight repair prompts for candidates that fail public tests.
2. Add a second selector signal, such as shortest-public-pass plus diversity or
   multi-candidate consensus on public tests.
3. Scale the same `n=4` and `n=8` protocol to the full 1,055-problem LCB set.
4. Record pass@1, public-pass rate, hidden-pass rate, wall time, and tokens so
   improvements are attributable to algorithmic test-time compute, not hidden
   leakage.
