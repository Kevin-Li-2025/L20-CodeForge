# Setup Report

Date: 2026-05-22

Remote host:

```text
ssh <user>@<l20-host>
project: ~/l20-codeforge
```

## Verified Hardware

```text
GPU: NVIDIA L20
VRAM visible to PyTorch: 44.52 GiB
Driver CUDA: 12.4
torch: 2.6.0+cu124
Python: 3.12.3
Disk after setup: 107G available on /
Project venv size: 5.7G
```

## Installed Stack

Core training stack:

```text
torch 2.6.0+cu124
transformers 5.9.0
trl 1.4.0
peft 0.19.1
bitsandbytes 0.49.2
datasets 4.8.5
```

Agent stack:

```text
mini-swe-agent 2.3.0
litellm 1.85.1
openai 2.38.0
tiktoken 0.13.0
```

## Verified Commands

```bash
cd /home/hhai/l20-codeforge
bash scripts/bootstrap_remote.sh
source .venv/bin/activate
python scripts/check_gpu.py
python -m l20_codeforge profile
pytest -q
ruff check .
mini --help
python -m l20_codeforge eval-card remote-smoke pass
python -m l20_codeforge smoke-loop
```

Results:

```text
GPU smoke: pass
pytest: 5 passed
ruff: all checks passed
mini CLI: available
eval card: artifacts/eval_cards/remote-smoke.json
smoke loop: generates executable repair trajectories and SFT JSONL
```

## Research Basis

Implementation choices are grounded in current primary sources:

- TRL GRPO supports custom reward functions and multiple reward components:
  https://huggingface.co/docs/trl/grpo_trainer
- mini-SWE-agent provides a simple bash-only, linear trajectory shape suitable
  for FT/RL:
  https://github.com/SWE-agent/mini-swe-agent
- SWE-bench uses Docker-based executable evaluation, so it should be introduced
  after local reward loops are stable:
  https://www.swebench.com/SWE-bench/installation/
- Qwen2.5-Coder-7B is a code-specific, Apache-2.0 model with repo-agent-relevant
  coding and long-context capability:
  https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- The Qwen2.5-Coder report describes the 0.5B/1.5B/3B/7B/14B/32B model family
  and code-focused training:
  https://arxiv.org/abs/2409.12186
- Unsloth's published VRAM table supports QLoRA as the default single-GPU path:
  https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- vLLM recommends a fresh uv environment because its binary stack is tightly
  coupled; keep it separate from training until inference scaling is needed:
  https://docs.vllm.ai/en/latest/getting_started/installation/gpu/

## Architecture Now In Place

```text
configs/
  l20_smoke.yaml
  qwen25_coder_7b_qlora.yaml
  grpo_guided_retry.yaml
src/l20_codeforge/
  context/compiler.py
  data/schema.py
  envs/repo_env.py
  evals/eval_card.py
  gpu/profile.py
  inference/selector.py
  rewards/patch_reward.py
  training/reward_functions.py
scripts/
  bootstrap_remote.sh
  check_gpu.py
  download_model.py
```

## Next Build Step

The next concrete step after the smoke loop is larger data and eval:

1. Create 20-50 small local repo repair tasks.
2. Run mini-SWE-agent trajectories into `artifacts/trajectories`.
3. Convert successful and partial trajectories into
   `data/processed/soft_verified_trajectories.jsonl`.
4. Run Qwen2.5-Coder-7B QLoRA SFT.
5. Add TRL GRPO only after reward decomposition catches obvious test hacking.
