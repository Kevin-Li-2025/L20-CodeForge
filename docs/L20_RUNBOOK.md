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
