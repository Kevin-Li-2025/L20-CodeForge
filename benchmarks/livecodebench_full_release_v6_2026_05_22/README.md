# LiveCodeBench Full `release_v6` Checkpoint

This directory records the first full LiveCodeBench code-generation run for
L20-CodeForge. It is a strict local evaluation over all 1,055
`livecodebench/code_generation_lite` `release_v6` tasks, graded with the
official LiveCodeBench evaluator.

## Result

| Model | Adapter | Decode | Samples | Tasks | Passed | pass@1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Qwen2.5-Coder-7B-Instruct` | none | greedy, `temperature=0` | 1 | 1,055 | 297 | 0.2815 |

This is not an official leaderboard submission. It is a reproducible local
checkpoint for a resource-constrained single-L20 setup. The official
leaderboard protocol may use different sampling and reporting settings; this
run intentionally uses greedy `n=1` to establish a low-cost full-suite floor.

## Reproducibility

- Benchmark: `livecodebench/code_generation_lite`, `release_v6`.
- Full task count: 1,055.
- Official evaluator repo commit:
  `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.
- Full JSONL SHA-256:
  `4519422b52d7dd243358ee721ecfe94e26ab364a7af2938c8e5d170a17bcadcd`.
- Prompt/public-only parquet SHA-256:
  `6a067f9b0762a2df9cad3e269872e189ce2d4f6ece7d8356a6f5d63cacdb5a96`.
- Saved generations SHA-256:
  `9d70473c14b20e6d3c514ff64d6f3525f870e5f25d18b55bd0ac83722912d3c5`.

The full hidden-test JSONL is not committed because it is 4.2GB. The committed
artifacts include saved generations, evaluator outputs, and compact summaries.

Source shards used to build the full JSONL:

| File | Rows | SHA-256 |
| --- | ---: | --- |
| `test.jsonl` | 400 | `2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f` |
| `test2.jsonl` | 111 | `095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3` |
| `test3.jsonl` | 101 | `28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79` |
| `test4.jsonl` | 101 | `d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da` |
| `test5.jsonl` | 167 | `7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214` |
| `test6.jsonl` | 175 | `bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5` |

## Generation/Evaluation Split

The remote L20 generated solutions from a prompt/public-only parquet. That file
contains problem prompts and public tests, but not hidden/private tests. The
saved generations were then pulled back and evaluated locally against the full
hidden-test JSONL with `codegen_metrics`.

Commands:

```bash
python scripts/run_lcb_subset_benchmark.py \
  --lcb-repo /path/to/LiveCodeBench \
  --lcb-commit 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 \
  --parquet data/raw/livecodebench/full_release_v6/release_v6_test_prompt_public_only.parquet \
  --output-dir benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_greedy_generate_only \
  --model /path/to/Qwen2.5-Coder-7B-Instruct \
  --n-samples 1 \
  --temperature 0 \
  --max-new-tokens 2048 \
  --generate-only \
  --overwrite

python scripts/evaluate_lcb_generations.py \
  --lcb-repo /path/to/LiveCodeBench \
  --lcb-commit 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 \
  --full-jsonl /path/to/release_v6_test_full.jsonl \
  --generations benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_greedy_generate_only/generations.json \
  --output-dir benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_greedy_full_eval \
  --k 1 \
  --max-samples 1 \
  --num-process-evaluate 8 \
  --timeout 8
```

## Breakdown

| Slice | Passed | Total | pass@1 |
| --- | ---: | ---: | ---: |
| Easy | 206 | 322 | 0.6398 |
| Medium | 82 | 383 | 0.2141 |
| Hard | 9 | 350 | 0.0257 |
| AtCoder | 146 | 602 | 0.2425 |
| Codeforces | 2 | 9 | 0.2222 |
| LeetCode | 149 | 444 | 0.3356 |

Failure classes:

| Class | Count |
| --- | ---: |
| Wrong answer | 605 |
| Runtime error | 85 |
| Time limit exceeded | 68 |

## Interpretation

The full score is much lower than the earlier 20-task subset result
(`12/20` greedy, `18/20` with `n=8` public-test selection), which means the
small subset was optimistic. The full-suite result is still valuable: it shows
where the L20-constrained strategy should focus next.

1. The model is usable on easy tasks but collapses on hard tasks.
2. Wrong answers dominate failures, so better reasoning/repair matters more
   than only optimizing runtime.
3. The earlier subset showed public-test selection can add large gains. The
   next full-suite experiment should run `n=4` or `n=8` generation with
   public-test selection and compare hidden pass@1 against this 0.2815 floor.

## Stratified `n=4` Public-Selection Probe

After the full greedy baseline, we ran a 60-task probe to test whether
test-time sampling and public-test selection still help on a harder slice of
the full distribution. The slice contains 20 easy, 20 medium, and 20 hard tasks,
chosen evenly by contest date within each difficulty.

| Run | Passed | Total | Score |
| --- | ---: | ---: | ---: |
| Full greedy baseline on same task IDs | 13 | 60 | 0.2167 |
| `temperature=0.8`, `n=4`, public-test selection | 21 | 60 | 0.3500 |
| `temperature=0.8`, `n=4`, hidden oracle | 23 | 60 | 0.3833 |

Breakdown for the selected run:

| Difficulty | Greedy | Public-selected `n=4` | Hidden oracle `n=4` |
| --- | ---: | ---: | ---: |
| Easy | 12/20 | 14/20 | 14/20 |
| Medium | 1/20 | 5/20 | 7/20 |
| Hard | 0/20 | 2/20 | 2/20 |

The selector recovered 21 of the 23 tasks that had at least one hidden-passing
candidate. This is the strongest current evidence that the next full run should
scale `n=4` public-test selection to all 1,055 tasks before spending time on
weight updates.

Primary artifacts:

- `analysis_summary.json`: compact score and breakdown.
- `stratified60_n4_public_select_summary.json`: `n=4` public-selection probe.
- `stratified60_selection_manifest.json`: sampled task IDs and data hash.
- `qwen25_coder_7b_greedy_generate_only/generations.json`: saved generated code.
- `qwen25_coder_7b_greedy_generate_only/report.json`: generation configuration.
- `qwen25_coder_7b_greedy_full_eval/report.json`: full hidden-test score.
- `qwen25_coder_7b_greedy_full_eval/eval_all.json`: per-task graded outputs.
