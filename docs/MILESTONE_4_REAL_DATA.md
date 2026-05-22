# Milestone 4: Real Data Ingestion

This milestone moves L20 CodeForge beyond synthetic repair tasks by adding a
real-world dataset registry and Hugging Face fetch path.

## Recommended Real Sources

Priority order:

1. `swe-bench-lite`  
   300 real Python GitHub issue/PR tasks in the current test split. Best first
   target for iteration.

2. `swe-bench-verified`  
   500 expert-verified SWE-bench tasks. Best evaluation-quality subset.

3. `swe-gym`  
   2,438 real-world Python SWE training instances with executable environments.

4. `swe-rebench-v2`  
   32,079 multilingual GitHub issue/PR tasks with Docker metadata.

5. `pybughive`, `bugsinpy`, `bugsjs`  
   Reproducible bug benchmarks that should be integrated through their own
   tooling rather than Hugging Face row loading.

## Commands

List known real sources:

```bash
python -m l20_codeforge list-real-sources
```

Fetch a small SWE-bench Lite sample:

```bash
python -m l20_codeforge fetch-real-tasks \
  swe-bench-lite \
  --output data/raw/real/swe_bench_lite_sample.jsonl \
  --limit 25
```

If the L20 host cannot reach Hugging Face, run the same command on a machine with
internet access and sync the generated JSONL/meta files into `data/raw/real/`.
The fetcher first tries `datasets.load_dataset`; if that fails, it falls back to
direct Hugging Face parquet downloads when `pyarrow` is available.

Fetch SWE-bench Verified:

```bash
python -m l20_codeforge fetch-real-tasks \
  swe-bench-verified \
  --output data/raw/real/swe_bench_verified.jsonl \
  --limit 500
```

Build real gold-patch SFT data:

```bash
python -m l20_codeforge build-real-sft \
  data/raw/real/swe_bench_lite_test.jsonl \
  --output data/processed/real_sft/swe_bench_lite_sft.jsonl
```

This uses real issue text and gold patches, but does not include `test_patch` in
the training prompt or completion.

Run a real-data SFT smoke on the L20 host:

```bash
python -m l20_codeforge train-real-sft \
  /home/hhai/model-cache/qwen2.5-0.5b-instruct \
  data/processed/real_sft/swe_bench_lite_sft.jsonl \
  --output-dir artifacts/checkpoints/qwen25-0p5b-real-sft-smoke \
  --max-steps 5 \
  --limit 64 \
  --max-length 2048
```

See `docs/MILESTONE_5_L20_7B_SFT.md` for the completed 7B QLoRA run on
SWE-bench Verified.

Fetch SWE-Gym:

```bash
python -m l20_codeforge fetch-real-tasks \
  swe-gym \
  --output data/raw/real/swe_gym_sample.jsonl \
  --limit 100
```

Fetch SWE-rebench-V2:

```bash
python -m l20_codeforge fetch-real-tasks \
  swe-rebench-v2 \
  --output data/raw/real/swe_rebench_v2_sample.jsonl \
  --limit 100
```

## Record Schema

Fetched rows are normalized into `RealTaskRecord`:

```text
dataset
split
instance_id
repo
base_commit
problem_statement
patch
test_patch
fail_to_pass
pass_to_pass
issue_url
pr_url
language
license
created_at
metadata
```

This is intentionally separate from local `TaskSpec`. Real SWE datasets require
checkout/build/evaluation harnesses before they can be run as local tasks.

## Research Sources

- SWE-bench dataset guide: https://www.swebench.com/SWE-bench/guides/datasets/
- SWE-Gym paper: https://arxiv.org/abs/2412.21139
- SWE-Gym dataset: https://huggingface.co/datasets/SWE-Gym/SWE-Gym
- SWE-rebench-V2 dataset: https://huggingface.co/datasets/nebius/SWE-rebench-V2
- PyBugHive: https://pybughive.github.io/
- BugsInPy: https://arxiv.org/abs/2401.15481
- BugsJS: https://bugsjs.github.io/

## Next Step

The next engineering task is a `materialize-real-task` command:

```text
RealTaskRecord
  -> clone repo at base_commit
  -> apply test_patch
  -> infer/run visible tests
  -> create local executable TaskSpec
```

That command should start with SWE-bench Lite because its schema is stable and
the official evaluation harness is mature.
