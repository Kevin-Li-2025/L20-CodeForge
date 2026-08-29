# Milestone 11: Base to Verified SFT to RLVR

Date: 2026-08-29

## Target

Run a matched model comparison on a newly frozen executable development set:

```text
Qwen2.5-Coder-7B-Instruct Base
  -> execution-verified self-SFT
  -> dense execution-reward GRPO/RLVR
```

The promotion gate is at least four additional greedy `n=1` passes on the
200-task development set (`+2.0` absolute points), with no HumanEval+ or MBPP+
pass@1 regression under official EvalPlus scoring. EvalPlus and LiveCodeBench
hidden tests are final guardrails, not training rewards or checkpoint-selection
signals.

## Frozen Data Receipt

The rStar-Coder `synthetic_rl_testcase` Dataset Server surface was used because
the original testcase directory is hundreds of gigabytes and the converted
surface provides a reproducible 4,000-row partial view. The campaign read the
first 1,600 rows and applied deterministic filters:

- 24 stdin/stdout tests per admitted task at most;
- at least eight tests after per-case input/output bounds;
- exact prompt deduplication;
- deterministic SHA-256 train/dev ordering with seed `20260829`;
- 5-gram Jaccard train-to-dev rejection at `0.85`;
- a separate comparison against all 1,055 committed LiveCodeBench v6 prompts.

Result: 1,434 admitted candidates, 800 train, 200 dev, and 166 rejected for too
few bounded tests. Train/dev hashes are pinned in the campaign config and data
manifest. The LiveCodeBench audit found zero exact matches, zero matches at or
above `0.85`, and a maximum similarity of `0.086486`.

## Verified SFT and RLVR Boundary

Verified SFT is self-training, not answer ingestion: Base samples four programs
per training prompt, the frozen executor runs them, and only programs passing
every admitted test may become assistant targets. RLVR starts from that SFT
adapter and trains only on prompt groups containing both passing and failing
Base rollouts, so GRPO has non-zero outcome variance.

The executor adds compile, timeout, address-space, output-size, and process-group
limits. It now also rejects common filesystem, subprocess, network, dynamic
execution, and dunder escape surfaces by AST before running model code. This is
defense in depth and is still not represented as a full container security
boundary.

## Current Execution State

- Remote Linux regression: `150 passed`.
- Four-shard GPU smoke job: `1491782`, all array tasks `COMPLETED`, exit `0:0`.
- Smoke result: `1/8` greedy passes and mean per-test pass fraction `0.489583`.
  This validates the execution path only and is not a model-quality claim.
- Formal Base dev job: `1491788`, four independent single-RTX-4090 shards.
- Verified-SFT rollout, SFT, RLVR, full dev comparison, and EvalPlus guardrails:
  pending at the time of this implementation receipt.

The exact live configuration is in
`configs/qwen25_coder_7b_base_sft_rlvr_20260829.yaml`. Machine-readable frozen
receipts are under `benchmarks/code_rlvr_base_sft_rlvr_2026_08_29/`.
