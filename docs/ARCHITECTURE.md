# Architecture

`l20-codeforge` is an eval-first coding-agent post-training system for a single
NVIDIA L20. Its job is to create a high-quality research surface before spending
GPU time.

## Control Loop

```text
task spec
  -> context compiler
  -> agent rollout or reference patch
  -> patch/test execution
  -> reward decomposition
  -> trajectory store
  -> SFT/DPO/GRPO dataset builders
  -> adapter training
  -> verifier-guided inference
  -> eval card
```

## Components

### Context

The context compiler creates a token-budgeted view of a repository. It favors:

1. files explicitly mentioned in the issue,
2. files with matching identifiers,
3. tests and configs,
4. recent failure logs.

This is a first-class target because repo-level agents fail under constrained
compute when they waste context on irrelevant files.

### Data

The data layer uses explicit schemas for:

- `TaskSpec`: the issue, repo, base commit, allowed commands, and hidden tests.
- `AgentStep`: command, observation, status, elapsed time, and token estimates.
- `Trajectory`: full rollout plus patch and reward decomposition.

The smoke task generator creates small Python repo-repair tasks with reference
patches. These are not meant to be impressive benchmarks. They are a regression
suite for the training data factory: if they fail, the larger SWE-style loop is
not ready.

This makes trajectories usable for SFT, DPO, GRPO, and offline audits.

### Reward

The reward is deliberately decomposed:

- executable tests,
- patch size and locality,
- anti-hack checks,
- self-verification quality.

The goal is not a single opaque score. The goal is a diagnosis that explains why
a candidate should be trained on, rejected, or inspected.

### Training

Single-L20 training should proceed in this order:

1. Soft-verified SFT on successful and partially successful trajectories.
2. DPO on critical action pairs.
3. Short-horizon GRPO/RLVR on tasks with cheap executable rewards.
4. Self-verification adapter or auxiliary data mixture.

### Inference

Inference is not one-shot. The default policy is:

1. sample several candidate patches,
2. run visible tests,
3. score patch locality and anti-hack signals,
4. ask for self-verification,
5. select the best candidate using a tournament selector.

This is where a small trained model can outperform its raw size.

## Environments

There are two environment levels:

- `RepoEnv`: local repo execution and patch evaluation.
- `SWE-bench` / mini-SWE-agent adapters: heavier executable benchmark paths.

The current scaffold implements the local repo path first because it is fast,
debuggable, and appropriate for L20 iteration.

## Current Executable Loop

```bash
python -m l20_codeforge smoke-loop
```

Outputs:

```text
artifacts/trajectories/smoke_reference.jsonl
data/processed/smoke_sft.jsonl
```

This loop validates patch application, test execution, reward scoring, trajectory
serialization, and SFT conversion without downloading a model.
