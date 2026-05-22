# Milestone 3: mini-SWE-agent Adapter

This milestone connects L20 CodeForge's executable repair suite to
mini-SWE-agent-style model rollouts.

## What Changed

The project can now:

1. export mini-SWE-agent prompts and suggested commands,
2. convert mini-SWE-agent `.traj.json` files into L20 CodeForge `Trajectory`,
3. extract unified diff patches from mini submissions or message outputs,
4. re-evaluate model patches with visible and hidden tests,
5. preserve mini command/observation steps,
6. build DPO-style chosen/rejected patch pairs.

## Commands

Generate the task suite and mini task records:

```bash
python -m l20_codeforge export-mini-tasks
```

This writes:

```text
artifacts/mini_swe/mini_task_records.jsonl
```

Each record contains:

- `task_id`
- `task_file`
- `repo`
- `prompt`
- `suggested_output`
- `suggested_command`

Convert a mini-SWE-agent trajectory:

```bash
python -m l20_codeforge convert-mini \
  data/raw/smoke_tasks/<task_id>/task.json \
  artifacts/mini_swe/trajectories/<task_id>.traj.json
```

Build preference pairs:

```bash
python -m l20_codeforge build-dpo \
  artifacts/trajectories/mini_swe_converted.jsonl \
  data/processed/preference_pairs.jsonl
```

## Patch Submission Contract

Generated task repositories include a helper:

```bash
python3 .l20_codeforge/make_patch.py
```

Agents should run that helper after editing source files and include its unified
diff in their final submission. The adapter will also search the message history
for a diff if the final submission is missing one.

## Why This Matters

Reference patches are useful for validating the data factory, but they are not
agent behavior. This adapter is the bridge from model interaction to training
data:

```text
mini-SWE trajectory
  -> patch extraction
  -> isolated patch eval
  -> visible/hidden reward
  -> L20 Trajectory JSONL
  -> SFT / DPO / GRPO data
```

## Verified

Local tests cover:

- mini action/observation extraction,
- fenced diff extraction,
- conversion from mini `.traj.json` to `Trajectory`,
- patch-helper output,
- mini task record export,
- DPO pair construction.

