# Milestone 9: X-Coder L20 +15 Probe

Date: 2026-05-24

## Goal

Focus the L20 run on a realistic path to exceed the Qwen2.5-Coder-7B reference by roughly +15 points on LiveCodeBench-style coding tasks without pretending that a tiny local LoRA has already improved the base model.

The strongest researched path is to use a verified-code-reasoning checkpoint/protocol as the teacher and benchmark target. X-Coder is directly relevant: the model card describes `IIGroup/X-Coder-RL-Qwen2.5-7B` as a Qwen2.5-Coder-derived RLVR/GRPO model trained on synthetic verified coding data, with recommended inference settings `temperature=0.6`, `top_p=0.95`, `top_k=20`, and `max_new_tokens=32768`. The X-Coder paper reports X-Coder-7B at `55.8 avg@8` on LiveCodeBench v6 and `62.9 avg@8` on v5, which is the clearest public signal that +15 over the original 7B coder is feasible under a comparable multi-sample protocol.

Related evidence: rStar-Coder reports that verified competitive-programming data lifts Qwen2.5-7B from `17.4%` to `57.3%` on LiveCodeBench. The common theme is not generic SFT. It is verified synthetic tasks, long reasoning, RL/post-training, and test-aware candidate selection.

Sources:

- X-Coder model card: https://huggingface.co/IIGroup/X-Coder-RL-Qwen2.5-7B
- X-Coder paper: https://arxiv.org/abs/2601.06953
- rStar-Coder paper: https://arxiv.org/abs/2505.21297

## Infra Changes

- Added bf16 CUDA loading for non-4bit inference via `device_map=auto`.
- Added `--attn-implementation` so benchmark reports capture SDPA/flash/eager choices.
- Added `--prompt-suffix` for controlled prompt experiments without editing code.
- Added `--question-ids` for precise failure reruns.
- Added `--stop-after-code-block` to stop generation once a complete fenced code block is emitted.

Verification:

- Local: `python3 -m pytest -q` passed with `108 passed` before the final stop-control patch.
- Local targeted after stop-control: `50 passed`.
- Remote targeted after stop-control: ruff passed and `50 passed`.

## Completed Probe Results

All runs used the real `release_v6` materialized LiveCodeBench data on the L20 host. The full JSONL SHA is `4519422b52d7dd243358ee721ecfe94e26ab364a7af2938c8e5d170a17bcadcd`.

### 8k bf16 final-code probe

Path:

- Generation report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_finalcode_stratified3_generate/report.json`
- Eval report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_finalcode_stratified3_eval/report.json`

Protocol:

- Model: `IIGroup/X-Coder-RL-Qwen2.5-7B`
- Prompt rendering: raw LCB prompt
- Prompt suffix: final Python solution only, fenced code block
- Sampling: `temperature=0.6`, `top_p=0.95`, `top_k=20`
- Runtime: bf16, SDPA, no 4-bit
- Max new tokens: 8192
- Tasks: first three stratified60 tasks: one easy, one medium, one hard

Result:

- Hidden eval: `2/3 = 66.7%`
- Easy passed.
- Medium passed.
- Hard failed due truncation/incorrect solution.
- Generation time: `415.623s`.

### 16k hard rerun

Path:

- Generation report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_16k_bf16_finalcode_hard1_generate/report.json`
- Eval report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_16k_bf16_finalcode_hard1_eval/report.json`

Protocol:

- Same model and decoding settings.
- Hard-only first task: `2784 power-of-heroes`.
- Max new tokens: 16384.

Result:

- Hidden eval: `0/1`.
- Generation time: `372.878s`.
- The output closed code blocks, so this was not just an 8k truncation issue. Single-sample longer reasoning did not rescue the hard problem.

## Current Interpretation

Good signal:

- X-Coder raw-prompt protocol is materially stronger than the local Qwen2.5-Coder-7B greedy baseline on the small mixed probe.
- bf16 is the correct L20 path: it uses roughly 15-16GB and keeps GPU utilization high; default 4-bit used roughly 6GB and was slower/less useful for this workload.
- 8k final-code is enough for easy/medium tasks in the initial mixed sample.

Bad/limiting signal:

- Hard competitive-programming tasks are not solved by simply increasing a single sample from 8k to 16k.
- Full 32k or 16k for every task is too slow on one L20.
- The scalable route has to be multi-sample search plus public/behavior/differential verification, with long budgets reserved for hard or failed cases.

## Next Run

Active remote run:

- `xcoder_rl_qwen25_7b_raw_topk20_16k_bf16_finalcode_stop_hard2784_n4bs1_publicselect`

Purpose:

- Test whether `n=4`, 16k, stop-after-code-block, and public-test selection can rescue the failed hard task.
- If it succeeds, use the same pattern for a 12-task mixed sample with 8k for easy/medium and 16k n=4 only for hard/failures.
- If it fails, prioritize better verifiers and model-generated tests over more token budget.

