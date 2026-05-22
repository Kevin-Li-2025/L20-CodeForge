# LiveCodeBench Lite Subset Benchmark

This directory records a strict execution benchmark against the official
LiveCodeBench evaluator. It is a fixed subset smoke benchmark, not a full
LiveCodeBench leaderboard run.

## Dataset

- Benchmark: `livecodebench/code_generation_lite`
- Release: `release_v6`
- Source shard: `test-00000-of-00009.parquet`
- Source shard SHA256: `e4b2a6aaee6ca2a1a8184f5baa204e6e2268578e8d2078aa962f1cca81f53eb0`
- Local subset SHA256: `9a35057e923ae1892c05202ca83ee6aeb47ec22911550e34c158f2b2a817b510`
- Subset rule: sort the source shard by compact public/private test payload size,
  then by `question_id`; keep the first 20 problems.
- LiveCodeBench repo commit: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`

The compact-test rule keeps the checkpoint fast enough for one L20 while still
using real LCB tasks and hidden tests. Treat this as an engineering signal for
prompting, inference, and selection methods, not as a claim about full LCB
performance.

## Results

| Run | Model | Decoding | Tasks | pass@1 | Passed | Generation sec | Eval sec |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen25_coder_7b_greedy_smoke3` | Qwen2.5-Coder-7B-Instruct | greedy | 3 | 0.333 | 1/3 | 48.133 | 0.534 |
| `qwen25_coder_7b_greedy_smalltests20` | Qwen2.5-Coder-7B-Instruct | greedy | 20 | 0.600 | 12/20 | 106.201 | 8.606 |
| `qwen25_coder_7b_temp08_n4_public_select_smoke3` | Qwen2.5-Coder-7B-Instruct | temp=0.8, n=4, public-select | 3 | 0.667 | 2/3 | 54.321 | 0.195 |
| `qwen25_coder_7b_temp08_n4_public_select_smalltests20` | Qwen2.5-Coder-7B-Instruct | temp=0.8, n=4, public-select | 20 | 0.850 | 17/20 | 227.834 | 8.562 |
| `qwen25_coder_7b_temp08_n8_public_select_smalltests20` | Qwen2.5-Coder-7B-Instruct | temp=0.8, n=8, public-select | 20 | 0.900 | 18/20 | 332.421 | 8.558 |

The `n=4` public-selection run evaluates all sampled candidates only on the
provided public tests, selects one candidate per problem, then grades only that
selected candidate with the full official hidden-test evaluator. On the 20-task
subset, public selection raised the strict score from `12/20` to `17/20`. The
public-test oracle was `19/20`, so one selected public-passing candidate still
failed hidden tests and one problem had no public-passing sample.

Increasing the sample budget to `n=8` raised the strict score to `18/20`. The
public-test oracle and selected public-pass counts were both `20/20`, but two
public-passing selections still failed hidden tests. That makes the next bottleneck
selection quality and public-to-hidden generalization, not merely candidate coverage.

## Files

- `*/generations.json`: prompts, raw model outputs, extracted code.
- `*/metrics.json`: `codegen_metrics` metrics/results plus sanitized evaluator metadata.
- `*/eval_all.json`: per-problem graded results with sanitized evaluator metadata.
- `*/report.json`: reproducibility metadata and summary metrics.
- `*/public_selection.json`: public-test selection metrics and selected candidate indices,
  present only for `--public-select` runs.

## Reproduce

```bash
python scripts/run_lcb_subset_benchmark.py \
  --lcb-repo /path/to/LiveCodeBench \
  --lcb-commit 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 \
  --parquet data/raw/livecodebench/release_v6_test_00000_smalltests20.parquet \
  --output-dir benchmarks/livecodebench_lite_subset_2026_05_22/qwen25_coder_7b_greedy_smalltests20 \
  --model /path/to/Qwen2.5-Coder-7B-Instruct \
  --n-samples 1 \
  --temperature 0 \
  --max-new-tokens 2048 \
  --num-process-evaluate 4 \
  --timeout 8 \
  --overwrite
```

Public-test selection run:

```bash
python scripts/run_lcb_subset_benchmark.py \
  --lcb-repo /path/to/LiveCodeBench \
  --lcb-commit 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 \
  --parquet data/raw/livecodebench/release_v6_test_00000_smalltests20.parquet \
  --output-dir benchmarks/livecodebench_lite_subset_2026_05_22/qwen25_coder_7b_temp08_n4_public_select_smalltests20 \
  --model /path/to/Qwen2.5-Coder-7B-Instruct \
  --n-samples 4 \
  --sample-batch-size 4 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 2048 \
  --public-select \
  --public-select-timeout 4 \
  --num-process-evaluate 4 \
  --timeout 8 \
  --overwrite
```
