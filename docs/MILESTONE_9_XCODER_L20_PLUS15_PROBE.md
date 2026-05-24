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
- Added `--response-prefix` for assistant-prefill experiments where the model input starts inside a Python code block and saved raw outputs are prepended with the same prefix before code extraction.
- Added `--response-prefix-mode starter-code` so assistant prefill can reuse each problem's exact starter signature instead of a fixed generic `class Solution`.
- Added `--static-retry-min-healthy-samples` and `--static-retry-max-extra-samples` for a public/hidden-free syntax and entrypoint retry gate.
- Added padding/trim handling for variable candidate counts before LiveCodeBench `codegen_metrics`, because upstream metadata grouping assumes each task has the same number of candidates.
- Added `scripts/regenerate_lcb_final_answers.py`, a second-pass regenerator that turns saved reasoning-heavy attempts into final executable code without using hidden expected outputs.
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

### mixed12 remaining-hard n=4 public selection

Path:

- Run report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_16k_bf16_finalcode_stop_mixed12_remaining_hard_n4_publicselect/report.json`
- Public selection: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_16k_bf16_finalcode_stop_mixed12_remaining_hard_n4_publicselect/public_selection.json`

Protocol:

- Run the two remaining hard tasks from the first-12 slice: `2879`, `3024`.
- `n_samples=4`, max new tokens 16384, stop-after-code-block, public-test selection.

Result:

- Hidden eval after public selection: `1/2 = 50.0%`.
- Rescued ID: `2879`.
- Still failed ID: `3024`.
- Public oracle/pass@4: `1/2 = 50.0%`.
- Generation time: `3155.956s`.
- Combining staged runs gives first-12 score `9/12 = 75.0%`: four easy/medium fast-stage passes, three failed-medium search rescues, hard `2784`, and hard `2879`.

### unresolved-medium strict-code n=4 public selection

Path:

- Run report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_strictcode_unresolved_medium_n4_publicselect/report.json`
- Public selection: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_strictcode_unresolved_medium_n4_publicselect/public_selection.json`

Protocol:

- Re-run unresolved medium tasks `2828` and `3166`.
- `n_samples=4`, max new tokens 8192, stop-after-code-block, public-test selection.
- Prompt suffix strengthened to require exactly one fenced Python code block and no other text.

Result:

- Hidden eval after public selection: `0/2`.
- Public oracle/pass@4: `0/2`.
- Generation time: `1453.078s`.
- The stricter prompt did not solve either unresolved medium task and still produced malformed/non-code-heavy candidates. This suggests prompt-only formatting pressure is not sufficient; these failures need better candidate generation or a code-repair/verifier pass.

### hard 3024 strict-code n=4 public selection

Path:

- Run report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_16k_bf16_strictcode_hard3024_n4_publicselect/report.json`
- Public selection: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_16k_bf16_strictcode_hard3024_n4_publicselect/public_selection.json`

Protocol:

- Re-run unresolved hard task `3024`.
- `n_samples=4`, max new tokens 16384, stop-after-code-block, public-test selection.
- Same strict code-only suffix used for unresolved medium retry.

Result:

- Hidden eval after public selection: `0/1`.
- Public oracle/pass@4: `0/1`.
- Generation time: `1726.398s`.
- This confirms prompt-only retries are not sufficient for the remaining first-12 failures.

### deterministic after-think extraction repair

Path:

- Failed medium repair report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_repair_extract_afterthink_failed_medium_finalcode_publicselect/repair_report.json`
- Failed medium eval report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_repair_extract_afterthink_failed_medium_finalcode_publicselect/eval/report.json`
- Remaining hard repair report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_repair_extract_afterthink_remaining_hard_finalcode_publicselect/repair_report.json`
- Remaining hard eval report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_repair_extract_afterthink_remaining_hard_finalcode_publicselect/eval/report.json`
- Hard 3024 strict repair report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_repair_extract_afterthink_hard3024_strict_publicselect/repair_report.json`
- Hard 3024 strict eval report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_repair_extract_afterthink_hard3024_strict_publicselect/eval/report.json`

Protocol:

- Fixed `strip_lcb_code_block` so `<answer>...</answer>` and `</think>` continuations are recursively passed through fenced-code extraction instead of returning prose plus code fences.
- Added `scripts/repair_lcb_generations.py` to deterministically re-extract saved candidates from `raw_outputs`, wrap top-level `def method(self, ...)` fragments into `class Solution`, and trim obvious malformed syntax tails when a compilable entrypoint remains.
- Repaired and re-evaluated the failed-medium, remaining-hard, and hard-3024 strict saved generations with public-test selection.

Result:

- Failed-medium final-code repair: hidden `3/5`, public oracle `3/5`, unchanged from the original failed-medium n=4 public-selection run.
- Remaining-hard final-code repair: hidden `1/2`, public oracle `1/2`, unchanged from the original remaining-hard n=4 public-selection run.
- Hard 3024 strict repair: hidden `0/1`, public oracle `0/1`, unchanged from the original strict-code run.
- The repair did identify and fix real extraction defects, especially prose retained after `</think>`, but the remaining first-12 misses are now dominated by algorithmic candidate failure rather than simple code-block extraction.

### independent unresolved-medium n=4 rerun

Path:

- Run report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_independent_unresolved_medium_n4_publicselect/report.json`
- Public selection: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_independent_unresolved_medium_n4_publicselect/public_selection.json`
- Repaired eval report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_repair_extract_afterthink_independent_unresolved_medium_n4_publicselect/eval/report.json`
- Repaired public selection: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_repair_extract_afterthink_independent_unresolved_medium_n4_publicselect/eval/public_selection.json`

Protocol:

- Re-ran unresolved medium tasks `2828` and `3166`.
- `n_samples=4`, max new tokens 8192, raw prompt rendering, bf16, SDPA, public-test selection.
- Prompt suffix told the model that the previous candidate set failed public tests and asked for an independent editorial-quality solution from first principles.
- Ran deterministic extraction repair on the saved generations and re-evaluated with public-test selection.

Result:

- Direct hidden eval after public selection: `0/2`.
- Direct public oracle/pass@4: `0/2`.
- Generation time: `1657.589s` for two medium tasks.
- Repaired hidden eval after public selection: `0/2`.
- Repaired public oracle/pass@4: `0/2`.
- The suffix increased reasoning length and cost but did not create a public-passing candidate. This is a negative result against continuing long independent reruns for these two medium failures.

### candidate-health audit and health-aware tie-breaker

Path:

- Candidate-health audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_unresolved_2026_05_24/audit.json`
- Independent medium health-tie recheck: `benchmarks/livecodebench_full_release_v6_2026_05_24/health_tiebreak_recheck_xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_independent_unresolved_medium_n4_publicselect/report.json`
- Repaired independent medium health-tie recheck: `benchmarks/livecodebench_full_release_v6_2026_05_24/health_tiebreak_recheck_xcoder_rl_repair_extract_afterthink_independent_unresolved_medium_n4_publicselect/report.json`
- Remaining hard health-tie recheck: `benchmarks/livecodebench_full_release_v6_2026_05_24/health_tiebreak_recheck_xcoder_rl_qwen25_7b_raw_topk20_16k_bf16_finalcode_stop_mixed12_remaining_hard_n4_publicselect/report.json`

Protocol:

- Added `scripts/audit_lcb_candidate_health.py` to summarize saved `generations.json`, `public_selection.json`, and sanitized eval metadata without using hidden expected outputs.
- The audit classifies syntax-valid candidates, entrypoint candidates, public-oracle availability, selected error metadata, and coarse failure modes.
- Updated public-selection tie-breaking so equal public scores prefer static-healthier candidates before applying `first`, `shortest`, or `longest`.
- Rechecked saved generations for independent medium, repaired independent medium, and remaining hard with the health-aware tie-breaker.

Result:

- Unresolved-run audit covered `6` runs, `14` task instances, and `56` candidates.
- Candidate syntax health was weak overall: `23/56 = 41.1%` syntax-valid and `32/56 = 57.1%` with an obvious entrypoint.
- The cleanest unresolved source remains the original failed-medium final-code run: `18/20` syntax-valid, `20/20` entrypoint candidates, but `2828` and `3166` still had public oracle `0/4`; those two are algorithmic failures under the current candidate set.
- Verbose/strict reruns degraded format quality: independent unresolved medium had only `1/8` syntax-valid candidates before repair, and `2/8` after repair.
- Health-aware tie-breaking changed selected candidate indices for `2828` in the independent/repaired-independent rechecks but did not change score: both remained `0/2`, public oracle `0/2`.
- Remaining hard recheck stayed `1/2`, public oracle `1/2`; `3024` still had no public-passing candidate.

### short-code and prefilled-code unresolved-medium n=8 probes

Path:

- Short-code run report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_2k_bf16_short_code_unresolved_medium_n8_publicselect/report.json`
- Short-code candidate-health audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_short_code_medium_2026_05_24/audit.json`
- Short-code repaired eval report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_repair_extract_afterthink_short_code_unresolved_medium_n8_publicselect/eval/report.json`
- Prefill-class run report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_2k_bf16_prefill_class_unresolved_medium_n8_publicselect/report.json`
- Prefill-class candidate-health audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_prefill_class_medium_2026_05_24/audit.json`

Protocol:

- Re-ran unresolved medium tasks `2828` and `3166`.
- Short-code probe used `n_samples=8`, max new tokens 2048, raw prompt rendering, bf16, SDPA, public-test selection, and a concise code-only suffix.
- Added and tested `--response-prefix`, then re-ran the same probe with the model prefixed into a Python block:
  `from typing import List`, `from collections import Counter`, and `class Solution:`.

Result:

- Short-code without prefill: hidden `0/2`, public oracle `0/2`, generation time `792.789s`.
- Short-code health audit: `0/16` syntax-valid and `0/16` entrypoint candidates. The 2048-token budget simply cut the model off before it produced usable code.
- Deterministic repair of the short-code run: hidden `0/2`, public oracle `0/2`, with no changed candidates.
- Prefill-class run: hidden `0/2`, public oracle `0/2`, generation time `791.091s`.
- Prefill-class health audit: `16/16` syntax-valid and `16/16` entrypoint candidates.
- This is a useful negative result: response prefixing fixes the format/entrypoint failure completely under short budgets, but `2828` and `3166` still need better algorithmic candidates or targeted verified data.

### starter-code prefill next3 n=1 held-out probe

Path:

- Run report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_2k_bf16_starter_prefill_next3_n1_publicselect/report.json`
- Candidate-health audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_starter_prefill_next3_2026_05_24/audit.json`

Protocol:

- Added `--response-prefix-mode starter-code`.
- Ran the next three stratified60 tasks not used in the first-12 staged result: `3329`, `3374`, and `3381`.
- Used the local Hugging Face snapshot path with offline mode after the first attempt stalled on network metadata before model load.
- Runtime: bf16, SDPA, raw prompt rendering, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_new_tokens=2048`, public-test selection with one candidate.
- Assistant prefill included a Python code fence, common imports, and each problem's exact starter code.

Result:

- Hidden/public-selected score: `2/3 = 66.7%`.
- Public oracle: `2/3 = 66.7%`.
- Generation time: `162.030s`.
- Candidate-health audit: `2/3` syntax-valid, `3/3` entrypoint candidates, and `1/3` selected syntax-error tasks.
- Interpretation: starter-code prefill preserves useful short-budget performance on a held-out slice and avoids generic entrypoint guessing, but it still needs a syntax/repair gate before scaling beyond small probes.

### static retry and 3329 budget/fallback probe

Path:

- Static-retry next3 report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_2k_bf16_starter_prefill_next3_n1_static_retry_publicselect/report.json`
- Static-retry next3 audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_starter_prefill_next3_static_retry_2026_05_24/audit.json`
- Body-only 3329 report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_1536_bf16_starter_prefill_bodyonly_3329_n1_static_retry_publicselect/report.json`
- 4k starter-prefill 3329 report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_4k_bf16_starter_prefill_3329_n1_publicselect/report.json`
- 8k starter-prefill 3329 report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_starter_prefill_3329_n1_publicselect/report.json`
- 8k raw final-code 3329 report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_8k_bf16_finalcode_3329_n1_publicselect/report.json`
- Combined candidate-health audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_3329_retry_budget_fallback_2026_05_24/audit.json`

Protocol:

- Added a static retry gate that only checks `ast.parse` and entrypoint presence; it does not inspect hidden expected outputs.
- Re-ran the held-out next3 with starter-code prefill, `n_samples=1`, `max_new_tokens=2048`, `static_retry_min_healthy_samples=1`, and `static_retry_max_extra_samples=2`.
- Fixed the runner after this exposed a LiveCodeBench assumption: upstream `codegen_metrics` asserts metadata length against the first task's candidate count, so variable-candidate runs now pad for metric evaluation and trim back to real candidate counts before public selection.
- Ran focused `3329` probes to test whether prompt wording, more short retries, budget escalation, or raw final-code fallback could recover the failed held-out task.

Result:

- Static-retry next3: hidden/public-selected `2/3`, public oracle `2/3`, generation time `215.755s`.
- Static-retry candidate counts: `[3, 1, 1]`; static-healthy candidate counts: `[0, 1, 1]`.
- Body-only `3329` at 1536 tokens with 2 extra retries: hidden/public `0/1`, public oracle `0/1`, `0/3` static-healthy.
- Starter-prefill `3329` at 4096 tokens with 1 extra retry: hidden/public `0/1`, public oracle `0/1`, `0/2` static-healthy.
- Starter-prefill `3329` at 8192 tokens: hidden/public `0/1`, public oracle `0/1`, `0/1` static-healthy.
- Raw final-code `3329` at 8192 tokens: hidden/public `0/1`, public oracle `0/1`, `0/1` static-healthy.
- Combined audit: `12` candidates across `7` task instances; only `2/12 = 16.7%` syntax-valid, both from the already-passing next3 tasks.
- Interpretation: static retry is useful as a gate and correctly surfaces failures, but `3329` is not fixed by more short samples, body-only prompting, or 8k budget. The model spends the generation on long reasoning/prose and never reaches a usable final implementation under these prompts.

### second-pass final-answer regeneration for 3329

Path:

- Regeneration report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_second_pass_final_answer_3329_from_raw8k_n2/report.json`
- Evaluation report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_second_pass_final_answer_3329_from_raw8k_n2_eval/report.json`
- Public selection: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_second_pass_final_answer_3329_from_raw8k_n2_eval/public_selection.json`
- Candidate-health audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_second_pass_3329_2026_05_24/audit.json`

Protocol:

- Used the saved raw 8k final-code attempt for `3329` as non-hidden source reasoning.
- Prompted the model to convert the previous reasoning into final executable code only.
- Used response prefix `````python``, `temperature=0.2`, `top_p=0.95`, `top_k=20`, `max_new_tokens=2048`, and generated two candidates.
- Evaluated with public-test selection and full release_v6 hidden replay.

Result:

- Regenerated candidates: `2`.
- Candidate health: `2/2` syntax-valid and `2/2` with entrypoint.
- Public oracle: `1/1`.
- Hidden/public-selected score: `1/1`.
- Regeneration time: `22.192s`.
- Interpretation: this rescues the held-out `3329` failure cleanly. The right escalation path is not more first-pass token budget; it is a second-pass final-answer conversion after the model has already done useful reasoning.

### second-pass first-12 remaining failures

Path:

- Medium failed-source regeneration report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_second_pass_final_answer_first12_medium_2828_3166_from_failed_n4/report.json`
- Medium failed-source evaluation report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_second_pass_final_answer_first12_medium_2828_3166_from_failed_n4_eval/report.json`
- Medium failed-source audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_second_pass_first12_medium_2026_05_24/audit.json`
- Hard 3024 regeneration report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_second_pass_final_answer_first12_hard3024_from_remaining_n4/report.json`
- Hard 3024 evaluation report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_second_pass_final_answer_first12_hard3024_from_remaining_n4_eval/report.json`
- Hard 3024 audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_second_pass_first12_hard3024_2026_05_24/audit.json`
- Medium independent-source regeneration report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_second_pass_final_answer_first12_medium_2828_3166_from_independent_n4/report.json`
- Medium independent-source evaluation report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_second_pass_final_answer_first12_medium_2828_3166_from_independent_n4_eval/report.json`
- Medium independent-source audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_second_pass_first12_medium_independent_2026_05_24/audit.json`

Protocol:

- Applied second-pass final-answer regeneration to the remaining first-12 failures: medium `2828`, medium `3166`, and hard `3024`.
- Medium retry 1 used the original failed-medium n=4 source reasoning; medium retry 2 used the independent unresolved-medium n=4 source reasoning for diversity.
- Hard retry used the remaining-hard n=4 source reasoning.
- All second-pass candidates were evaluated with public-test selection before full release_v6 hidden replay.

Result:

- Medium failed-source second pass: generated `8` candidates in `423.804s`; hidden/public-selected `0/2`, public oracle `0/2`; audit showed `8/8` syntax-valid and `8/8` with entrypoint.
- Medium independent-source second pass: generated `8` candidates in `423.274s`; hidden/public-selected `0/2`, public oracle `0/2`; audit showed `8/8` syntax-valid and `8/8` with entrypoint.
- Hard `3024` second pass: generated `4` candidates in `258.789s`; public oracle `1/1`, public selection pass rate `1/4`, and hidden/public-selected `1/1`; audit showed `3/4` syntax-valid and `4/4` with entrypoint.
- Updated first-12 staged score: `10/12 = 83.3%`. The remaining unsolved first-12 tasks are `2828` and `3166`.
- Interpretation: second-pass regeneration is valuable for reasoning-to-code collapse and rescued hard `3024`, but it does not fix the two medium algorithmic failures. Those need targeted verified data, stronger algorithmic teacher traces, or a different search distribution.

### targeted teacher trace and code-prefix rescue for 2828/3166

Path:

- Targeted teacher trace source: `data/processed/livecodebench/targeted_teacher_traces_2828_3166_v1.json`
- Natural-language trace regeneration report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_second_pass_teacher_trace_2828_3166_n4/report.json`
- Natural-language trace evaluation report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_second_pass_teacher_trace_2828_3166_n4_eval/report.json`
- Natural-language trace audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_teacher_trace_2828_3166_2026_05_24/audit.json`
- Targeted code-prefix `2828` regeneration report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_targeted_code_prefix_2828_n4/report.json`
- Targeted code-prefix `3166` regeneration report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_targeted_code_prefix_3166_n4/report.json`
- Targeted code-prefix combined generations: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_targeted_code_prefix_2828_3166_n4_combined/generations.json`
- Targeted code-prefix combined evaluation report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_targeted_code_prefix_2828_3166_n4_combined_eval/report.json`
- Targeted code-prefix audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_targeted_code_prefix_2828_3166_2026_05_24/audit.json`

Protocol:

- Built two short verified traces from the public problem statements only: one greedy string trace for `2828`, and one frequency/base-size trace for `3166`.
- First tested natural-language second-pass regeneration with `n=4` per task and the same public-selection/full-replay evaluator.
- Because natural-language trace still expanded into comment-heavy non-solutions, tested a narrower code-prefix completion probe: the prompt supplied the key function skeleton and left the model to complete the implementation.

Result:

- Natural-language trace: generated `8` candidates in `74.876s`; `8/8` syntax-valid and `8/8` with entrypoint, but public oracle `0/2` and hidden/public-selected `0/2`.
- Code-prefix `2828`: generated `4` candidates in `22.275s`.
- Code-prefix `3166`: generated `4` candidates in `19.755s`.
- Code-prefix combined evaluation: public oracle `2/2`, public-selected public pass `2/2`, hidden/public-selected `2/2`.
- Code-prefix audit: `8/8` syntax-valid, `8/8` with entrypoint, public oracle task rate `1.0`, hidden-selected task rate `1.0`.
- Interpretation: the remaining first-12 failures can be rescued by targeted verified code-prefix distillation. This is not a clean broad benchmark claim, because the prefix is task-targeted, but it is a strong signal for the next scalable method: mine or generate concise verified code-prefix traces, train a small adapter on those patterns, and validate on less curated held-out LCB slices.
- Staged first-12 status: strict prompt/search route is `10/12 = 83.3%`; targeted code-prefix rescue reaches `12/12 = 100%` on the first-12 probe with the overfitting caveat above.

### medium control12 automatic starter-prefix generalization check

Path:

- Run report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_2k_bf16_starter_prefill_medium_control12_n1_publicselect/report.json`
- Public-selection payload: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_2k_bf16_starter_prefill_medium_control12_n1_publicselect/public_selection.json`
- Metrics payload: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_2k_bf16_starter_prefill_medium_control12_n1_publicselect/metrics.json`
- Candidate-health audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_starter_prefill_medium_control12_2026_05_24/audit.json`

Protocol:

- Selected the remaining `12` medium tasks from the existing stratified60 shard that were not part of the first-12 or held-out next3 probes: `3416`, `3475`, `3639`, `3657`, `3751`, `3786`, `abc320_c`, `abc331_c`, `abc364_c`, `abc370_d`, `abc375_c`, `abc400_d`.
- Used automatic starter-code/code-fence response prefixing only; no per-task algorithm prefix and no hidden-output tuning.
- Ran X-Coder bf16 with `n=1`, temperature `0.6`, top-p `0.95`, top-k `20`, `max_new_tokens=2048`, public-test selection, and one static-health retry per task.

Result:

- Generated `17` total candidates for `12` tasks in `836.485s` because `5` tasks needed the extra static retry.
- Public oracle: `0/12`; public-selected hidden replay: `0/12`.
- Candidate health: `9/17` syntax-valid, `17/17` with an entrypoint, and `3` selected candidates were syntax-invalid.
- Failure split: `3` tasks had no syntax-valid candidates after retry; the remaining `9` had syntax-valid candidates rejected by public tests.
- Interpretation: automatic starter-prefixing alone does not generalize from the first-12 targeted rescue. The next credible improvement must be an automatic verified-prefix/trace generator or a trained adapter evaluated on this control12 slice before claiming broad generalization.

### medium control12 strict starter-prefix check

Path:

- Run report: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_1k_bf16_strict_starter_prefill_medium_control12_n1_publicselect/report.json`
- Public-selection payload: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_1k_bf16_strict_starter_prefill_medium_control12_n1_publicselect/public_selection.json`
- Metrics payload: `benchmarks/livecodebench_full_release_v6_2026_05_24/xcoder_rl_qwen25_7b_raw_topk20_1k_bf16_strict_starter_prefill_medium_control12_n1_publicselect/metrics.json`
- Candidate-health audit: `benchmarks/livecodebench_full_release_v6_2026_05_24/lcb_candidate_health_strict_starter_prefill_medium_control12_2026_05_24/audit.json`

Protocol:

- Reused the same medium control12 slice.
- Kept automatic starter-code/code-fence prefixing and public-test selection, but added a strict implementation suffix: compact executable code only, no reasoning/prose/comments, and close the code block after final code.
- Reduced budget from `2048` to `1024` new tokens and lowered temperature from `0.6` to `0.2`.

Result:

- Generated `14` candidates for `12` tasks in `353.427s`, down from `17` candidates in `836.485s`.
- Candidate health improved from `9/17 = 52.9%` syntax-valid to `10/14 = 71.4%` syntax-valid; entrypoint stayed `100%`.
- Public oracle remained `0/12`; public-selected hidden replay remained `0/12`.
- Failure split: `2` tasks still had no syntax-valid candidates, and `10` tasks had syntax-valid candidates rejected by public tests.
- Interpretation: strict output control reduces wasted L20 time and syntax collapse, but it does not solve algorithmic generalization. The control12 gate now strongly argues for automatic verified algorithm-prefix generation or training, not more prompt hygiene.

## Current Interpretation

Good signal:

- X-Coder raw-prompt protocol is materially stronger than the local Qwen2.5-Coder-7B greedy baseline on the small mixed probe.
- bf16 is the correct L20 path: it uses roughly 15-16GB and keeps GPU utilization high; default 4-bit used roughly 6GB and was slower/less useful for this workload.
- 8k final-code is enough for easy/medium tasks in the initial mixed sample.
- The failed hard problem was rescued by `n=4` plus public-test selection.
- The failed-medium rerun rescued three of five medium failures, confirming that candidate search helps beyond hard tasks.
- The first-12 staged protocol reached `9/12 = 75.0%`, far above the original 7B greedy baseline on the full release_v6 run.
- The extraction repair is useful infrastructure for future runs because it prevents reasoning/prose leakage from being misclassified as model algorithm failure.
- Candidate-health audit now gives a lightweight gate to avoid mistaking syntax/entrypoint collapse for algorithmic weakness.
- Response prefixing is the right infra path for short-budget candidate generation: it changed `0/16` syntax-valid candidates into `16/16` syntax-valid candidates on the unresolved-medium probe.
- Starter-code response prefixing is better than a fixed generic class prefill for broader LCB slices because it uses the benchmark's actual function signature per task.
- Static retry is now implemented correctly for variable-candidate runs and avoids hidden-output leakage.
- Second-pass final-answer regeneration is the first method in this cycle that converted a reasoning-heavy failure into a public/hidden passing solution.
- Second-pass also rescued hard `3024`, moving the first-12 staged score from `9/12` to `10/12`.
- Targeted verified code-prefix completion rescued the two remaining first-12 medium failures (`2828`, `3166`) with public and hidden replay both passing.
- The best current architecture is now clearer: long reasoning/search finds ideas; a short verified code-prefix or distilled trace is needed to force executable implementation.
- Strict starter-prefix prompting makes medium-control generation cheaper and healthier: `353s` vs `836s`, and syntax-valid rate `71.4%` vs `52.9%`.

Bad/limiting signal:

- Hard competitive-programming tasks are not solved by simply increasing a single sample from 8k to 16k.
- Full 32k or 16k for every task is too slow on one L20.
- The scalable route has to be multi-sample search plus public/behavior/differential verification, with long budgets reserved for hard or failed cases.
- Hard `n=4` at 16k took about 22.7 minutes for one task, so the next benchmark must be staged by difficulty.
- Single-sample medium coverage is weak in the first mixed12 slice; only `4/9` easy/medium tasks passed.
- Medium `n=4` at 8k is also expensive: five tasks took about 57.2 minutes and still left two failures.
- Public-selection search still failed three first-12 tasks: medium `2828`, medium `3166`, and hard `3024`. These need stricter output-format control and/or better verification, not just more samples.
- The strict-code prompt failed to rescue `2828` and `3166`, so the remaining medium failures likely need a repair/verifier stage rather than another prompt-only rerun.
- The strict-code prompt also failed to rescue hard `3024`, with public pass@4 still zero.
- Deterministic extraction repair did not improve the first-12 score beyond `9/12`; next gains require new candidates, stronger public/differential tests, or targeted post-training data rather than more cleanup of the same samples.
- A long independent-rerun suffix also failed on unresolved medium tasks with public oracle `0/2`, so the next attempt should not spend more L20 time on the same style of verbose prompt.
- The remaining medium failures are not fixed by more verbose prompting; the original clean-code candidates fail public tests, while later verbose candidates often fail syntax/entrypoint health.
- Even with response-prefix forcing and healthy syntax, unresolved medium public oracle stayed `0/2`; this is now clearly an algorithm/data issue, not just output formatting.
- Starter-code prefill still produced one syntax-invalid selected candidate on the held-out next3 slice, so short-budget prefill needs deterministic repair or a syntax-aware retry loop before a larger run.
- The `3329` failure is more severe than syntax alone: starter-prefill, body-only prompting, 4k/8k budget escalation, and raw final-code fallback all failed to produce a syntax-valid candidate. This needs a different final-answer forcing method or targeted verified data.
- The second pass was proven on one held-out task only; it must be validated on more failures before claiming broad generalization.
- Medium `2828` and `3166` remain unsolved after two independent second-pass source pools; their failure mode is now algorithmic, not syntax or final-answer extraction.
- Natural-language teacher traces alone did not rescue `2828` or `3166`: they produced syntactically valid comment-heavy candidates with public oracle `0/2`.
- The targeted code-prefix rescue is intentionally narrow and should be treated as a distillation/probing result, not as a clean LiveCodeBench leaderboard score.
- A fresh medium control12 slice scored `0/12` under automatic starter-code/code-fence prefixing, so the first-12 targeted rescue currently does not transfer without a learned or generated task-specific algorithm prefix.
- Strict output control also scored `0/12` on the same medium control12 slice. It fixes part of the format problem, but not the algorithm-selection problem.

## Next Run

Active remote run:

- None. The L20 is idle after the targeted code-prefix evaluation.

Purpose:

- Next step is to generalize the code-prefix result beyond the hand-targeted first-12 slice.
- Use deterministic extraction repair by default for future saved generations, but do not spend more cycles cleaning the same candidates unless a public-test signal changes.
- Prefer short, constrained code-only candidate construction or targeted verified training examples over long free-form reasoning reruns.
- Run candidate-health audit before committing expensive reruns; if syntax-valid/public-oracle remains zero, switch strategy instead of increasing `n`.
- For future short-budget generation, use `--response-prefix`; do not use short `max_new_tokens` without prefill.
- Prefer `--response-prefix-mode starter-code` over a fixed `class Solution` prefix for mixed held-out slices.
- Do not scale the current starter-prefill prompt blindly; route syntax-collapse tasks like `3329` to second-pass final-answer regeneration or targeted teacher-data construction.
- Build a small set of mined/verified code-prefix traces for similar greedy-string and frequency-partition tasks, then evaluate on a held-out medium slice without per-task hand prefixes.
- Keep two separate headlines: strict prompt/search route `10/12 = 83.3%`; targeted code-prefix probe `12/12 = 100%` with overfitting caveat.
- Use the medium control12 run as the immediate generalization gate: any automatic prefix generator or adapter should first improve public oracle above `0/12` on this slice before spending time on larger LCB runs.
- Default future short-budget control runs to the strict suffix because it is faster and healthier, but do not expect it to improve benchmark score without additional algorithm-prefix signal.
