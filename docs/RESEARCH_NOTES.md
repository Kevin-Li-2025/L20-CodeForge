# Research Notes

This is the implementation-facing summary of the current research.

## Tooling Choices

### TRL First

Hugging Face TRL exposes GRPO with custom reward functions, including multiple
reward functions and async callables. That matches our need to combine tests,
patch quality, anti-hack checks, and self-verification without building a full
RL system immediately.

Source: https://huggingface.co/docs/trl/grpo_trainer

### mini-SWE-agent For Trajectory Shape

mini-SWE-agent is intentionally simple: bash-only actions and linear histories.
That is useful for post-training because the trajectory can be serialized almost
directly into model messages.

Source: https://github.com/SWE-agent/mini-swe-agent

### SWE-bench As Heavy Eval, Not First Dependency

SWE-bench is Docker-based and appropriate for serious evals. The first scaffold
uses local repo execution so reward bugs are visible before expensive benchmark
runs.

Source: https://www.swebench.com/SWE-bench/installation/

### Qwen2.5-Coder-7B As Primary Base

Qwen2.5-Coder-7B has code-specific pretraining/post-training, permissive
licensing, and a 128K context window. It is a good L20-scale base for adapter
training and repo-level coding experiments.

Sources:

- https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- https://arxiv.org/abs/2409.12186

### Unsloth/QLoRA For Single-GPU Efficiency

Unsloth's public guidance indicates 7B QLoRA can fit in very small VRAM and 14B
QLoRA is still far below the L20's 46GB memory. That makes QLoRA the default
adaptation path.

Source: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements

### vLLM As Separate Inference Environment

vLLM recommends a fresh uv-managed environment because the wheel bundles a
validated PyTorch/binary stack. This project keeps vLLM optional and separate to
avoid breaking training dependencies.

Source: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/

## Research Bet

The credible novelty is not "trained another coder model." It is:

```text
resource-constrained coding-agent post-training with explicit context omission,
soft verification, decomposed executable rewards, and verifier-guided inference.
```

This can produce useful, publishable artifacts:

- a clean single-GPU training recipe,
- a trajectory schema for coding agents,
- an ablation on context omission,
- reward-hacking diagnostics for patch generation,
- a small but reliable local SWE-style benchmark.

