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

## 2026 Research Update: What The Regression Means

The current L20 run improved held-out assistant-token NLL/perplexity, but failed
the first executable SWE task. That is a weak or negative capability signal, not
a success metric. It means the adapter became better at matching the gold-patch
distribution under teacher forcing, while the free-running patch policy still
fails at file localization, diff validity, or test-passing repair.

Recent SWE-agent research points in the same direction:

- SWE-bench Verified emphasizes executable `FAIL_TO_PASS` tests and human
  verification because ordinary issue/patch datasets can be underspecified or
  unfair. Source:
  https://openai.com/index/introducing-swe-bench-verified/
- SWE-Gym reports large gains from training agents and verifiers on executable
  real-world task environments, not only next-token patch imitation. Source:
  https://arxiv.org/abs/2412.21139
- SWE-RL reports that RL on software-evolution data can improve SWE-bench
  Verified solve rate, while an SFT baseline degraded on average across
  out-of-domain reasoning tasks. Source:
  https://arxiv.org/abs/2502.18449
- DeepSeek-R1 uses rule-based rewards for coding/math/logical reasoning and
  explicitly relies on compiler/test-style feedback where correctness is
  objectively checkable. Source:
  https://www.nature.com/articles/s41586-025-09422-z
- 2026 code-RL work warns that softer pass-rate rewards can reduce sparsity but
  may not reliably beat binary rewards in controlled experiments. Source:
  https://arxiv.org/abs/2605.02944
- 2026 reward-hacking work argues that visible tests alone are insufficient for
  long-horizon coding agents; held-out composition tests are needed to measure
  whether the agent built the intended system. Source:
  https://arxiv.org/abs/2605.21384

Practical conclusion for this repo:

```text
NLL down + executable pass rate flat/down = stop scaling SFT, improve eval and reward.
```

The next high-quality architecture should be:

1. Real executable eval first: base fail, gold pass, candidate pass/fail.
2. Patch-format reward: valid unified diff, applies cleanly, touches plausible
   files, avoids test edits unless requested.
3. Execution reward: run `FAIL_TO_PASS`; then add `PASS_TO_PASS` or held-out
   smoke tests to catch regressions.
4. Verifier-guided sampling: generate several candidates, filter by apply/test,
   then train preference pairs from pass/fail attempts.
5. Only then run GRPO/DPO-style post-training, with the L20 reserved for compact
   QLoRA adapters and CPU/disk reserved for repo materialization.

## Hiring Signal Fit

The public role signal is strongly aligned with this direction:

- OpenAI's agentic post-training roles call out LLM/RL/post-training/evals,
  graders, reward models, data pipelines, coding agents, tool use, and
  end-to-end ownership. Source:
  https://openai.com/careers/researcher-agentic-post-training-san-francisco/
- Anthropic's production post-training role emphasizes the full post-training
  stack, RLHF/alignment methods, distributed systems, complex ML systems, and
  debugging model training. Source:
  https://www.anthropic.com/careers/jobs/4613592008
- Jane Street ML roles emphasize empirical ML, noisy nonstationary data,
  training/inference infrastructure, hyperparameter work, distributed training
  performance, RL/LLMs, and GPU clusters. Source:
  https://www.janestreet.com/join-jane-street/machine-learning/
- Citadel/Citadel Securities roles emphasize rigorous ML research, high-quality
  code, large/unstructured datasets, backtesting, AI infrastructure, and
  accelerated compute. Sources:
  https://www.citadel.com/careers/details/machine-learning-researcher-phd-graduate-us/
  and https://www.citadelsecurities.com/careers/details/research-engineer/

Therefore the portfolio target should not be a leaderboard-only fine-tune. It
should be a compact research system that demonstrates judgment: real data,
execution-grounded evals, reproducible negative results, GPU-aware training, and
clear next experiments.
