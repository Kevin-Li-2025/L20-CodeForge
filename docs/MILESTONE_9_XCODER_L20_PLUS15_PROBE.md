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
- Added per-batch checkpointing so long `n>1` searches write `generations.json` after each completed candidate batch instead of only after the full problem finishes.

Verification:

- Local: `python3 -m pytest -q` passed with `108 passed` before the final stop-control patch.
- Local targeted after stop-control: `50 passed`.
- Remote targeted after stop-control: ruff passed and `50 passed`.
- Local and remote targeted after per-batch checkpointing: `51 passed`; remote ruff passed.

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

### 16k hard n=4 public selection

Path:

- Run report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_16k_bf16_finalcode_stop_hard2784_n4bs1_publicselect_ckpt/report.json`
- Public selection: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_16k_bf16_finalcode_stop_hard2784_n4bs1_publicselect_ckpt/public_selection.json`

Protocol:

- Same model and decoding settings.
- Hard task: `2784 power-of-heroes`.
- `n_samples=4`, `sample_batch_size=1`, public-test selection, max new tokens 16384.
- `--stop-after-code-block` enabled.

Result:

- Hidden eval after public selection: `1/1`.
- Public-test candidate pass distribution: one of four candidates passed public tests; selected index `1`.
- Public selection metrics: `pass@1=0.25`, `pass@4=1.0`.
- Generation time: `1362.501s`.
- This is the first strong evidence that the path for hard problems is multi-sample verified search, not single-sample longer reasoning.

### mixed12 easy/medium n=1 fast stage

Path:

- Run report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_finalcode_stop_mixed12_easy_medium_n1/report.json`
- Public selection: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_finalcode_stop_mixed12_easy_medium_n1/public_selection.json`

Protocol:

- First twelve stratified60 tasks, excluding the three hard tasks.
- `n_samples=1`, max new tokens 8192, public-selection enabled but only one candidate exists.

Result:

- Hidden eval: `4/9 = 44.4%`.
- Passed IDs: `2777`, `2855`, `3242`, `3324`.
- Failed IDs: `2779`, `2828`, `2916`, `3166`, `3240`.
- Generation time: `1269.858s`.
- This invalidates the assumption that easy/medium can all run as cheap single-sample tasks; medium tasks also need candidate search or stronger extraction/prompting.

### mixed12 failed-medium n=4 public selection

Path:

- Run report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_finalcode_stop_mixed12_failed_medium_n4_publicselect/report.json`
- Public selection: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_finalcode_stop_mixed12_failed_medium_n4_publicselect/public_selection.json`

Protocol:

- Re-run the five failed medium tasks from the fast stage: `2779`, `2828`, `2916`, `3166`, `3240`.
- `n_samples=4`, max new tokens 8192, stop-after-code-block, public-test selection.

Result:

- Hidden eval after public selection: `3/5 = 60.0%`.
- Rescued IDs: `2779`, `2916`, `3240`.
- Still failed IDs: `2828`, `3166`.
- Public oracle/pass@4: `3/5 = 60.0%`.
- Generation time: `3434.152s`.
- Combined with the four already-passing easy/medium tasks, the first-12 easy/medium subtotal is now `7/9`.

## Current Interpretation

Good signal:

- X-Coder raw-prompt protocol is materially stronger than the local Qwen2.5-Coder-7B greedy baseline on the small mixed probe.
- bf16 is the correct L20 path: it uses roughly 15-16GB and keeps GPU utilization high; default 4-bit used roughly 6GB and was slower/less useful for this workload.
- 8k final-code is enough for easy/medium tasks in the initial mixed sample.
- The failed hard problem was rescued by `n=4` plus public-test selection.
- The failed-medium rerun rescued three of five medium failures, confirming that candidate search helps beyond hard tasks.

Bad/limiting signal:

- Hard competitive-programming tasks are not solved by simply increasing a single sample from 8k to 16k.
- Full 32k or 16k for every task is too slow on one L20.
- The scalable route has to be multi-sample search plus public/behavior/differential verification, with long budgets reserved for hard or failed cases.
- Hard `n=4` at 16k took about 22.7 minutes for one task, so the next benchmark must be staged by difficulty.
- Single-sample medium coverage is weak in the first mixed12 slice; only `4/9` easy/medium tasks passed.
- Medium `n=4` at 8k is also expensive: five tasks took about 57.2 minutes and still left two failures.

## Next Run

Active remote run:

- `xcoder_rl_qwen25_7b_raw_topk20_16k_bf16_finalcode_stop_mixed12_remaining_hard_n4_publicselect`

Purpose:

- Run the two remaining hard tasks in the first-12 slice: `2879`, `3024`.
- Use `n=4`, max new tokens 16384, stop-after-code-block, and public-test selection.
- Combine: four easy/medium fast passes + three medium search rescues + q2784 hard rescue + remaining hard results.
