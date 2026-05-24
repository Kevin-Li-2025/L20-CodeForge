# Reproducibility

This document defines the exact local checks and benchmark-artifact verification
used for the current L20-CodeForge README claims.

The repository keeps benchmark summaries, saved generations, evaluator outputs,
and hashes under version control. Large private LiveCodeBench payloads are not
committed; they must be materialized locally from the benchmark source when a
full hidden replay is required.

## Environment

Recommended local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,bench]"
```

Expected package install outcome:

```text
Successfully installed l20-codeforge
```

The GitHub Actions CI uses Python `3.11` and installs the same `.[dev,bench]`
extras before running tests and scorecard checks.

## CI Check

Command:

```bash
python -m pytest -q
```

Expected output shape:

```text
135 passed in <time>s
```

The exact wall time depends on machine load. The pass count is the important
invariant for this snapshot.

## Static L20 Profile

Command:

```bash
python -m l20_codeforge profile
```

Expected output contains:

```json
{
  "gpu": "NVIDIA L20",
  "vram_gb": 48,
  "preferred_base_models": [
    "Qwen/Qwen2.5-Coder-7B-Instruct"
  ]
}
```

## Generalization Scorecard

Command:

```bash
python scripts/build_generalization_scorecard.py \
  --output-dir /tmp/l20_codeforge_scorecard_check
```

Expected output:

```json
{
  "status": "PASS",
  "checks": [
    {
      "name": "lcb_overall_improves",
      "value": 0.100474,
      "threshold": 0.0,
      "passed": true
    }
  ]
}
```

The actual output includes all slice checks. For the committed snapshot, every
check must have `"passed": true`.

## Benchmark Artifact Hashes

Verify the committed artifact hashes:

```bash
shasum -a 256 \
  benchmarks/generalization_scorecard_2026_05_23/scorecard.json \
  benchmarks/livecodebench_full_release_v6_2026_05_22/full_n8_public_select_summary.json \
  benchmarks/evalplus_l20_codeforge_2026_05_22/summary.csv \
  benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n8_public_select_full_eval/report.json \
  benchmarks/evalplus_l20_codeforge_2026_05_22/rechecks/manifest.json
```

Expected output:

```text
1eb0402378ea25732225b29d7ba367b6111ab3351e54cc7c01fa7646a7a12712  benchmarks/generalization_scorecard_2026_05_23/scorecard.json
2a0ff919aa15eb9ecdf74824f7bf790a23f6d0197ef74970b6190c60e0e00772  benchmarks/livecodebench_full_release_v6_2026_05_22/full_n8_public_select_summary.json
08732bbb76450f92ef3c02fa97a163aba01f71028365072c205c5a3af45d5550  benchmarks/evalplus_l20_codeforge_2026_05_22/summary.csv
7272f5591c2f868c059226a2a5ec8fc772994cfafd20eb8397a2b6d90aed64bf  benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n8_public_select_full_eval/report.json
e86db2af864a9c8896dcd1bc2d4d7b44af7fa395b856ea02b6f0e69c31c915cc  benchmarks/evalplus_l20_codeforge_2026_05_22/rechecks/manifest.json
```

The CI workflow verifies the three top-level claim artifacts on every push and
pull request:

- `benchmarks/generalization_scorecard_2026_05_23/scorecard.json`
- `benchmarks/livecodebench_full_release_v6_2026_05_22/full_n8_public_select_summary.json`
- `benchmarks/evalplus_l20_codeforge_2026_05_22/summary.csv`

## LiveCodeBench Full Replay Boundary

The full hidden-test JSONL is intentionally not committed. To materialize it
locally:

```bash
python scripts/materialize_lcb_release_jsonl.py \
  --release-version release_v6 \
  --output-jsonl data/raw/livecodebench/full_release_v6/release_v6_test_full.jsonl \
  --manifest data/raw/livecodebench/full_release_v6/release_v6_test_full.manifest.json
```

Expected committed summary for the current headline result:

```json
{
  "benchmark": "livecodebench/code_generation_lite release_v6",
  "tasks": 1055,
  "score": {
    "passed": 403,
    "total": 1055,
    "pass_at_1": 0.3819905213270142
  },
  "baseline_greedy": {
    "passed": 297,
    "total": 1055,
    "pass_at_1": 0.28151658767772514
  }
}
```

The full command sequence for generation and hidden replay is documented in:

```text
benchmarks/livecodebench_full_release_v6_2026_05_22/README.md
```

## EvalPlus Rechecks

HumanEval+:

```bash
python -m l20_codeforge eval-evalplus humaneval \
  benchmarks/evalplus_l20_codeforge_2026_05_22/samples/humaneval.mixed-target.literal-combined.public-consensus-selected.samples.jsonl \
  --output /tmp/humaneval_recheck.json \
  --parallel 8
```

Expected committed result:

```text
base pass@1 0.982
plus pass@1 0.927
```

MBPP+:

```bash
python -m l20_codeforge eval-evalplus mbpp \
  benchmarks/evalplus_l20_codeforge_2026_05_22/samples/mbpp.temp08.n5-plus-basefallback-n30.public-consensus-shortest-selected.samples.jsonl \
  --output /tmp/mbpp_recheck.json \
  --parallel 8
```

Expected committed result:

```text
base pass@1 0.960
plus pass@1 0.817
```

## Clean-Room Claim Rules

- Public tests and public examples may be used for selection and repair.
- Hidden/private tests may only be used for final replay and audit.
- Do not report targeted probes as broad benchmark results.
- Do not update benchmark headline claims without updating the scorecard and
  artifact hashes in this document.
