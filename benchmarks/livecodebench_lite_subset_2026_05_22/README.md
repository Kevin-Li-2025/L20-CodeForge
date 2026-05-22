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

## Files

- `*/generations.json`: prompts, raw model outputs, extracted code.
- `*/metrics.json`: `codegen_metrics` metrics/results plus sanitized evaluator metadata.
- `*/eval_all.json`: per-problem graded results with sanitized evaluator metadata.
- `*/report.json`: reproducibility metadata and summary metrics.

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
