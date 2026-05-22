# Milestone 2: 36-Task Repair Suite

This milestone expands the data factory from a tiny smoke test into a broader
single-file repair suite.

## Scope

The suite now contains 36 Python repair tasks across:

```text
aggregation
boundary handling
control flow
dict operations
formatting
graph algorithms
matrix utilities
metrics
normalization
parsing
privacy
ranking
sequence handling
sorting
time parsing
URL encoding
```

Each task has:

- a buggy source file,
- a reference patch,
- visible tests exposed through `TaskSpec.visible_test_command`,
- hidden tests exposed through `TaskSpec.hidden_test_command`,
- tags for report grouping.

## Command

```bash
python -m l20_codeforge smoke-loop
```

Expected result:

```json
{
  "tasks": 36,
  "success": 36,
  "status_counts": {
    "success": 36
  },
  "sft_records": 36
}
```

## Generated Artifacts

```text
data/raw/smoke_tasks/
artifacts/trajectories/smoke_reference.jsonl
artifacts/reports/smoke_reference_report.json
data/processed/smoke_sft.jsonl
```

## Why This Matters

The suite is still synthetic, but it is deliberately varied. It catches common
repair-agent failure modes:

- off-by-one loops,
- wrong denominator or zero division,
- falsy-value bugs,
- order preservation,
- missing validation,
- delimiter handling,
- hidden edge cases,
- reward hacking against visible tests only.

This is the right substrate before introducing mini-SWE-agent model rollouts:
the task and reward machinery can now be tested without spending API or GPU
budget.

## Next Step

Run model-generated patches against the same suite:

```bash
mini --task "$(cat data/raw/smoke_tasks/<task>/task.json)" --yolo --output artifacts/trajectories/<task>.json
```

That command shape will need a thin adapter so mini-SWE-agent trajectories can be
normalized into L20 CodeForge's `Trajectory` schema.

