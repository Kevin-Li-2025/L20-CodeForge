# L20 Runbook

## Hardware Policy

Observed host:

```text
GPU: NVIDIA L20, 46068 MiB
Driver: 550.163.01
CUDA from driver: 12.4
RAM: about 15 GiB
Disk after cleanup: about 112 GiB free
```

The GPU is usable. The system RAM is the practical bottleneck.

## Environment Policy

Use one lightweight research environment first:

```bash
cd ~/l20-codeforge
bash scripts/bootstrap_remote.sh
source .venv/bin/activate
python scripts/check_gpu.py
pytest -q
ruff check .
python -m l20_codeforge smoke-loop
```

Keep vLLM in a separate environment if needed. The vLLM docs recommend a fresh
environment because its wheel bundles a tightly pinned binary stack.

## Disk Policy

Use these locations:

```text
~/l20-codeforge/artifacts      checkpoints, eval cards, generated data
~/l20-codeforge/data           local processed data only
~/.cache/huggingface           Hugging Face cache
```

Do not put model weights in the git tree. Do not create per-project 6GB venvs for
every small experiment.

## First Experiments

1. GPU and import smoke:

```bash
python scripts/check_gpu.py
python -m l20_codeforge profile
pytest -q
```

2. Local reward smoke on a small repository:

```bash
python -m l20_codeforge init-dirs
python -m l20_codeforge eval-card local-reward-smoke pass
python -m l20_codeforge smoke-loop
```

3. Download primary model only when the stack is ready:

```bash
python scripts/download_model.py Qwen/Qwen2.5-Coder-7B-Instruct
```

## Training Budget

For Qwen2.5-Coder-7B:

- QLoRA: expected to fit comfortably.
- LoRA bf16: possible but not the default because RAM and sequence length matter.
- GRPO: keep `num_generations` small, start at 4, and disable vLLM until the
  basic reward loop is stable.

For 14B:

- Use QLoRA only.
- Avoid long-context GRPO until 7B results justify it.

## Smoke Loop Expectations

`smoke-loop` should report 36 tasks and 36 successful reference patches.
It writes:

```text
artifacts/trajectories/smoke_reference.jsonl
artifacts/reports/smoke_reference_report.json
data/processed/smoke_sft.jsonl
```

If this fails, fix the local data/eval/reward loop before downloading large
weights or starting GPU training.

For repeat validation:

```bash
bash scripts/run_remote_quality_gate.sh
```

## Agent Rollout Bridge

Export mini-SWE-agent prompts:

```bash
python -m l20_codeforge export-mini-tasks
```

After running mini and saving a `.traj.json`, normalize it:

```bash
python -m l20_codeforge convert-mini \
  data/raw/smoke_tasks/<task_id>/task.json \
  artifacts/mini_swe/trajectories/<task_id>.traj.json
```

Then build preference pairs when the trajectory file contains both strong and
weak patches for the same task:

```bash
python -m l20_codeforge build-dpo artifacts/trajectories/mini_swe_converted.jsonl
```

## Real Data

List supported real datasets:

```bash
python -m l20_codeforge list-real-sources
```

Fetch a small real sample first:

```bash
python -m l20_codeforge fetch-real-tasks \
  swe-bench-lite \
  --output data/raw/real/swe_bench_lite_sample.jsonl \
  --limit 25
```

Keep these JSONL files out of git. They are generated data.

## Real SFT Smoke

Use the local cached 0.5B model first to verify the training stack:

```bash
python -m l20_codeforge train-real-sft \
  /home/hhai/model-cache/qwen2.5-0.5b-instruct \
  data/processed/real_sft/swe_bench_lite_sft.jsonl \
  --output-dir artifacts/checkpoints/qwen25-0p5b-real-sft-smoke \
  --max-steps 5 \
  --limit 64 \
  --max-length 2048
```

For a 7B offline run, first stage the model directory on the L20 host. If the
host cannot reach Hugging Face directly, use the mirror endpoint and explicit
filenames:

```bash
HF_ENDPOINT=https://hf-mirror.com hf download \
  Qwen/Qwen2.5-Coder-7B-Instruct \
  config.json generation_config.json model.safetensors.index.json \
  model-00001-of-00004.safetensors \
  model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors \
  model-00004-of-00004.safetensors \
  tokenizer.json tokenizer_config.json merges.txt vocab.json \
  --local-dir /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct
```

Then run the 7B QLoRA smoke:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m l20_codeforge train-real-sft \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  data/processed/real_sft/swe_bench_lite_sft.jsonl \
  --output-dir artifacts/checkpoints/qwen25-coder-7b-real-sft-smoke \
  --max-steps 5 \
  --limit 64 \
  --max-length 2048 \
  --gradient-accumulation-steps 4 \
  --load-in-4bit
```

Use this verified-data configuration for the first full single-L20 pass:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m l20_codeforge train-real-sft \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  data/processed/real_sft/swe_bench_verified_sft.jsonl \
  --output-dir artifacts/checkpoints/qwen25-coder-7b-verified-sft-4096-epoch1 \
  --max-steps 64 \
  --limit 500 \
  --max-length 4096 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --load-in-4bit
```

Observed on the L20: 64 steps over 500 SWE-bench Verified SFT records completed
in 1011.2 seconds with train loss 0.9147. Peak memory was about 29.6 GiB. A
4096-token batch size of 4 OOMed near 41.9 GiB, so use batch size 2 unless the
loss path is changed to avoid full-logit materialization.

## Real Executable Patch Eval

Likelihood eval is not enough for SWE-style coding. Use `eval-real-patch` to
validate the harness on every candidate patch:

For this Django task, install the historical repo's minimal runtime dependencies
in the active evaluation venv:

```bash
python -m pip install pytz sqlparse
```

```bash
python -m l20_codeforge eval-real-patch \
  data/raw/real/swe_bench_lite_test.jsonl \
  django__django-10924 \
  artifacts/real_eval/patches/django-10924-adapter-generated-2048.patch \
  --output artifacts/real_eval/reports/django-10924-adapter-eval.json \
  --repos-dir artifacts/real_eval/repos \
  --test-command 'PYTHONPATH=. python tests/runtests.py model_fields.test_filepathfield.FilePathFieldTests.test_callable_path --verbosity=2 --noinput' \
  --candidate-name adapter-2048 \
  --timeout-seconds 120
```

The command checks three states from the same base commit:

```text
base + test_patch: expected fail
gold + test_patch: expected pass
candidate + test_patch: target pass
```

Treat `base=failed` and `gold=passed` as the minimum harness sanity check. If
the candidate fails while SFT likelihood improved, the signal is not "training
worked"; it is "the model learned patch distribution but not executable repair."
