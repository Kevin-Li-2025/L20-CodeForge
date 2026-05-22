# Milestone 1: Local Data Factory

This milestone turns L20 CodeForge from a scaffold into a runnable data factory.

## What It Does

```bash
python -m l20_codeforge smoke-loop
```

The command:

1. generates three small Python repo-repair tasks,
2. writes a reference patch for each task,
3. applies each patch in an isolated worktree,
4. runs `python3 -m unittest discover -s tests`,
5. decomposes reward into tests, patch quality, anti-hack, and self-verification,
6. appends trajectories to JSONL,
7. converts successful trajectories into chat SFT JSONL.

## Why This Matters

This is the minimum credible post-training loop. It proves that the project can
produce executable, inspectable training examples before using GPU time.

## Outputs

```text
data/raw/smoke_tasks/
artifacts/trajectories/smoke_reference.jsonl
data/processed/smoke_sft.jsonl
```

These paths are intentionally ignored by git. Generated tasks, trajectories,
checkpoints, and model weights should not be committed.

## Quality Gate

Expected local tests:

```bash
pytest -q
ruff check .
```

Expected remote smoke:

```bash
cd ~/l20-codeforge
source .venv/bin/activate
python -m l20_codeforge smoke-loop
```

Only after this passes should the project move to mini-SWE-agent rollouts and
Qwen2.5-Coder-7B QLoRA SFT.
