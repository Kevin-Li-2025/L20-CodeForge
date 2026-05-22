# l20-codeforge

Resource-constrained coding-agent RL and post-training stack for a single NVIDIA L20.

The design target is not broad pretraining. It is a compact loop that turns limited
GPU time into measurable software-engineering capability:

1. Build executable repo-level evaluation first.
2. Collect and normalize agent trajectories.
3. Train small adapters with soft-verified SFT and preference data.
4. Run guided GRPO/RLVR only on short, high-signal tasks.
5. Use self-verification and verifier-guided candidate selection at inference time.

## Why This Shape

The stack follows current evidence from:

- TRL GRPO custom reward support:
  https://huggingface.co/docs/trl/grpo_trainer
- mini-SWE-agent's simple bash-only trajectory shape:
  https://github.com/SWE-agent/mini-swe-agent
- SWE-bench's Docker-based executable evaluation:
  https://www.swebench.com/SWE-bench/installation/
- Qwen2.5-Coder model family and long-context code capabilities:
  https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- Unsloth VRAM guidance for QLoRA/LoRA:
  https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- vLLM's recommendation for fresh uv-managed environments:
  https://docs.vllm.ai/en/latest/getting_started/installation/gpu/

## Repository Layout

```text
configs/                 L20-first experiment configs
docs/                    architecture and runbooks
scripts/                 setup, smoke, and model utility scripts
src/l20_codeforge/
  context/               repo context packing and omission policies
  data/                  task and trajectory schemas
  envs/                  local repo execution adapters
  evals/                 eval cards and reporting
  gpu/                   L20 profile and memory policy
  inference/             candidate verification and selection
  rewards/               executable and patch-quality rewards
  training/              TRL-compatible reward functions and config helpers
tests/                   lightweight unit tests
```

## First Remote Setup

On the L20 host:

```bash
cd ~/l20-codeforge
bash scripts/bootstrap_remote.sh
source .venv/bin/activate
python scripts/check_gpu.py
pytest -q
```

The bootstrap script installs uv if missing, creates a Python 3.12 venv, installs a
CUDA 12.4 PyTorch wheel, then installs this package with training and dev
dependencies. It does not download large model weights.

## Model Strategy

Primary training target:

- `Qwen/Qwen2.5-Coder-7B-Instruct`

L20 policy:

- Use QLoRA for 7B/14B work.
- Use short-context GRPO first: 2K-4K completion budgets, small groups.
- Keep vLLM in a separate environment if needed, because vLLM pins binary stacks.
- Keep all generated checkpoints under `artifacts/`, never under the repo root.

## Smoke Gates

Before real training, these must pass:

```bash
python scripts/check_gpu.py
python -m l20_codeforge profile
pytest -q
```

After that, run a toy reward loop before loading a large model:

```bash
python -m l20_codeforge init-dirs
python -m l20_codeforge eval-card smoke pass
python -m l20_codeforge smoke-loop
```

`smoke-loop` creates 36 executable repo-repair tasks with visible and hidden
tests, evaluates known-good patches in isolated worktrees, writes trajectories,
builds a report, and builds chat SFT JSONL:

```text
data/raw/smoke_tasks/
artifacts/trajectories/smoke_reference.jsonl
artifacts/reports/smoke_reference_report.json
data/processed/smoke_sft.jsonl
```

This is the first quality gate. Do not start GPU training until this local loop
passes on the target machine.

Milestone notes:

- `docs/MILESTONE_1_DATA_FACTORY.md`: first executable data loop.
- `docs/MILESTONE_2_REPAIR_SUITE.md`: 36-task visible/hidden repair suite.
- `docs/MILESTONE_3_MINI_SWE_ADAPTER.md`: mini-SWE-agent trajectory adapter and DPO pairs.
- `docs/MILESTONE_4_REAL_DATA.md`: real SWE dataset registry and fetch path.

Agent trajectory bridge:

```bash
python -m l20_codeforge export-mini-tasks
python -m l20_codeforge convert-mini data/raw/smoke_tasks/<task>/task.json artifacts/mini_swe/trajectories/<task>.traj.json
python -m l20_codeforge build-dpo artifacts/trajectories/mini_swe_converted.jsonl
```

Real data entry:

```bash
python -m l20_codeforge list-real-sources
python -m l20_codeforge fetch-real-tasks swe-bench-lite --output data/raw/real/swe_bench_lite_sample.jsonl --limit 25
```
