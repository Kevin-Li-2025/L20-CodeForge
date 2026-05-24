# Milestone 7: LiveCodeBench Strict Subset and Full Run

The project now has a second public benchmark line beyond EvalPlus. The new
runner uses local HF inference on the L20 and delegates grading to the official
LiveCodeBench `codegen_metrics` evaluator.

## Why This Matters

EvalPlus is useful for function-level robustness, but it is not enough to show
general contest-style coding ability. LiveCodeBench adds newer problems,
stdin/stdout tasks, hidden tests, and LeetCode-style functional tasks. This is a
better proxy for whether post-training and test-time compute methods transfer
outside HumanEval/MBPP.

## Current Checkpoint

- Benchmark: `livecodebench/code_generation_lite`, `release_v6`.
- Scope: fixed 20-problem compact-test subset from shard
  `test-00000-of-00009.parquet`.
- Official evaluator repo commit:
  `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.
- Model: `Qwen2.5-Coder-7B-Instruct`, 4-bit NF4, bf16 compute, greedy decoding.
- Result: `12/20`, pass@1 `0.600`.
- Runtime: `106.201s` generation, `8.606s` evaluation.

The first test-time compute run uses temperature `0.8`, `n=4`, and public-test
selection:

- Result: `17/20`, pass@1 `0.850`.
- Runtime: `227.834s` generation, `2.230s` public selection, `8.562s` final
  evaluation.
- Public-test oracle: `19/20`.
- Residual failures: `2754`, `2837`, `3017`.

Increasing the same setup to `n=8`:

- Result: `18/20`, pass@1 `0.900`.
- Runtime: `332.421s` generation, `5.271s` public selection, `8.558s` final
  evaluation.
- Public-test oracle: `20/20`.
- Residual failures: `2754`, `3017`.

This is not a leaderboard score. The subset is deliberately small so a single
L20 can run frequent checkpoints. The value is the strict hidden-test signal:
we can now measure whether future public-test selection, repair, and RL/SFT
changes improve or regress on real LCB tasks.

## Full `release_v6` Checkpoint

A full-suite greedy baseline is now recorded under
`benchmarks/livecodebench_full_release_v6_2026_05_22/`.

- Scope: all 1,055 `release_v6` tasks.
- Evaluation: saved generations graded locally with official LiveCodeBench
  hidden/private tests.
- Model: `Qwen2.5-Coder-7B-Instruct`, 4-bit NF4, bf16 compute, no adapter.
- Decode: greedy, `n=1`, `temperature=0`, `max_new_tokens=2048`.
- Result: `297/1055`, pass@1 `0.2815`.
- Generation runtime: `6795.360s`.
- Evaluation runtime: `467.664s`.
- Full JSONL SHA-256:
  `4519422b52d7dd243358ee721ecfe94e26ab364a7af2938c8e5d170a17bcadcd`.

Breakdown:

- Easy: `206/322`, pass@1 `0.6398`.
- Medium: `82/383`, pass@1 `0.2141`.
- Hard: `9/350`, pass@1 `0.0257`.
- AtCoder: `146/602`, pass@1 `0.2425`.
- LeetCode: `149/444`, pass@1 `0.3356`.

This is a stricter and more reliable baseline than the 20-task subset. The
subset result was useful for fast iteration, but it was optimistic; the full
run shows that medium/hard contest reasoning is the real bottleneck.

## Immediate Diagnosis

The 3-problem smoke run found two real model failures:

- `1873_A`: the model overcomplicated a simple swap rule and failed sample/hidden
  behavior.
- `2754`: the model mishandled negative-product parity.

The 20-problem run recovered to `0.600` pass@1, and public-test selection lifted
that to `0.850` with `n=4` and `0.900` with `n=8`. This is a strong algorithmic
signal on the subset, but the full greedy score of `0.2815` makes the caveat
clear: the small subset cannot be treated as a public benchmark score.

On the full run, failures are mostly wrong answers:

- Wrong answer: `605`.
- Runtime error: `85`.
- Time limit exceeded: `68`.

That points to repair, verifier-guided selection, and curriculum targeted at
medium/hard algorithmic reasoning before spending GPU time on broader SFT.

## `n=4` Selection Probe

The first follow-up test used a 60-task stratified slice from the full
`release_v6` set: 20 easy, 20 medium, and 20 hard tasks, sampled evenly by
contest date within each difficulty.

- Greedy full-run baseline on the same 60 task IDs: `13/60`, pass@1 `0.2167`.
- `temperature=0.8`, `n=4`, public-test selection: `21/60`, score `0.3500`.
- Hidden oracle over the same four candidates: `23/60`, score `0.3833`.
- Public selector captured `21/23` tasks that had a hidden-passing candidate.

By difficulty:

- Easy: greedy `12/20`, selected `14/20`, oracle `14/20`.
- Medium: greedy `1/20`, selected `5/20`, oracle `7/20`.
- Hard: greedy `0/20`, selected `2/20`, oracle `2/20`.

This is a good sign for the L20 strategy. Sampling plus public-test selection
found real hidden-test improvements, and most remaining misses came from not
having a correct candidate rather than selecting the wrong candidate. The next
high-value benchmark was full `n=4` public selection over all 1,055 tasks.

## Full `n=4` Public-Selection Result

The full `n=4` run is now recorded under
`benchmarks/livecodebench_full_release_v6_2026_05_22/`.

- Scope: all 1,055 `release_v6` tasks.
- Model: `Qwen2.5-Coder-7B-Instruct`, 4-bit NF4, bf16 compute, no adapter.
- Decode: `temperature=0.8`, `top_p=0.95`, `n=4`, `max_new_tokens=2048`.
- Selection: public-test score, tie-broken by shortest selected code.
- Result: `378/1055`, pass@1 `0.3583`.
- Gain over greedy full baseline: `+81` tasks, `+0.0768` absolute pass@1.
- Generation runtime: `34489.714s`.
- Hidden evaluation runtime after reusing saved public selection: `706.327s`.
- Saved generations SHA-256:
  `3293b100575d483dfa33e79a75f3048d1edfa4c62847150f5e4f19ae480d0b0d`.

Breakdown:

- Easy: `251/322`, pass@1 `0.7795`.
- Medium: `110/383`, pass@1 `0.2872`.
- Hard: `17/350`, pass@1 `0.0486`.
- AtCoder: `183/602`, pass@1 `0.3040`.
- LeetCode: `190/444`, pass@1 `0.4279`.

Public tests selected a public-passing candidate on `546/1055` tasks, but only
`378/1055` of those selected programs passed hidden tests. That is a clear
engineering signal: public-test selection is worth keeping, but public tests
alone are too weak as the final verifier. The next step should target
public-pass-hidden-fail cases with repair, additional generated tests, or a
second candidate-ranking signal.

## Behavior-Consensus Selector Probe

A first verifier extension now runs each candidate on deterministic mutations of
the public inputs, then selects by public-test score, behavior-cluster size,
per-test consensus, behavior success count, and the configured tie-breaker.

On the 60-task stratified slice:

- Public-test selection: `21/60`, pass@1 `0.3500`.
- Public + deterministic behavior consensus: `21/60`, pass@1 `0.3500`.
- The selector changed `8/60` choices but produced no hidden-score change.
- Two hidden-oracle misses had all candidates tied on public tests; deterministic
  mutations did not distinguish the hidden-passing candidate.

This is a useful negative result. Simple public-input perturbation is not enough
for the main failure mode. The next verifier must generate candidate-aware
distinguishing inputs, not only mutate visible examples.

A second run exercised the new public-score reuse path:

- Same 60-task slice, same deterministic behavior inputs: `21/60`, pass@1
  `0.3500`.
- Public scores loaded from the saved `public_selection.json` instead of
  re-running public tests.
- Behavior-selection wall time dropped from `152.367s` to `86.522s`.

The project also generated a 64-task candidate-aware prompt bank from the full
`n=4` public-selection result. These prompts target public-score-tied tasks and
ask a local model to produce additional input-only differential tests. The prompt
bank intentionally contains no hidden test outputs.

## Generated Behavior-Test Probe

The first local-model generation pass over that prompt bank is recorded under
`benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_candidate_aware_behavior_tests64/`.

- Prompt records: `64`.
- Raw model outputs: `64`.
- Parsed behavior-test records: `54`.
- Generated behavior inputs: `558`.
- Generation wall time on the L20: `580.994s`.

On the 54 tasks with parsed behavior inputs:

- Reused public-test selection: `51/54`, pass@1 `0.9444`.
- Public score plus generated behavior consensus: `52/54`, pass@1 `0.9630`.
- Conservative public-pass behavior override with consensus margin `1`:
  `52/54`, pass@1 `0.9630`.
- Net targeted gain: `+1` task.

The selector changed three tasks. Two remained hidden-pass cases with a
different selected candidate (`2837`, `2848`), and one changed a hidden failure
into a pass (`2854`, `decremental-string-concatenation`). This is the first
positive signal that candidate-aware generated tests can add useful selection
information beyond public tests.

The conservative override policy is now implemented in
`scripts/evaluate_lcb_generations.py` as
`--behavior-selection-policy conservative-public-pass`. The stricter replay
required the replacement candidate to pass public tests, have at least `6`
generated behavior tests, reach behavior success rate `>=0.9`, and beat the
public-selected candidate by at least `1` behavior-consensus point. The three
overrides still survived that filter, including the real `2854` improvement.

A full-suite hybrid replay replaced those 54 records in the prior public
selection payload and produced `377/1055` in one run. That number is not adopted
as the main score: the diff showed two unrelated non-target tasks with unchanged
selected indices (`abc363_c`, `abc378_e`) flipping from pass to fail, and both
passed again in an immediate two-task recheck. Until full-suite evaluation has a
retry-stabilized path, the stable full score remains the prior public-selection
result, `378/1055`.

`scripts/compare_lcb_selection_runs.py` now provides an explicit audit gate for
this situation. It marks the 54-task conservative replay as `improved` and the
full hybrid replay as `unstable_replay`, because `abc363_c` and `abc378_e` had
unchanged selected code but different hidden outcomes.

The same audit gate can now consume candidate recheck payloads for those
unchanged-code flips. Applying the two-task recheck changes the full hybrid
audit from raw `377/1055` to stabilized `379/1055`, with status
`stabilized_improved` and no unresolved unchanged-code flips. This is an audit
count, not the headline benchmark score; the next official score should come
from an evaluator run that performs this retry/majority logic automatically.

The next gate is therefore conservative:

- Build retry or majority-recheck into full-suite evaluation rather than only
  post-hoc audit.
- Apply the conservative override policy to full-suite replay.
- Refresh the generalization scorecard only after the stabilized replay has no
  unresolved unchanged-code outcome flips.

## Research Update: High-Leverage Next Step

The current literature points strongly toward generated tests and execution
grounding as the most promising path for a single-L20 setup:

- CodeT uses generated tests plus dual execution agreement to select among
  multiple code samples:
  https://arxiv.org/abs/2207.10397
- AlphaCodium shows that code generation improves when organized as a
  test-based, multi-stage flow rather than a single prompt:
  https://arxiv.org/abs/2401.08500
- S* adds test-time scaling for code by generating distinguishing inputs for
  pairwise candidate comparison:
  https://arxiv.org/abs/2502.14382
- HardTests shows that high-quality synthetic edge-case tests materially improve
  verifier precision and recall, especially on hard programming tasks:
  https://arxiv.org/abs/2505.24098
- ACECODER uses automated test-case synthesis to build preference/reward data
  and reports gains for Qwen2.5-Coder-7B-Instruct:
  https://aclanthology.org/2025.acl-long.587/
- rStar-Coder reports large gains from verified competitive-programming data
  with rich test cases and long-reasoning solutions:
  https://arxiv.org/abs/2505.21297

The project now has the start of this path:

- `scripts/evaluate_lcb_generations.py` can consume external behavior-test
  inputs through `--behavior-inputs`.
- The same evaluator can reuse saved public scores via `--behavior-public-scores`
  so full-suite behavior selection does not repeat public-test execution.
- `scripts/build_lcb_behavior_test_prompts.py` builds candidate-aware prompts
  for public-score-tied tasks and can parse LLM JSON outputs into evaluator-ready
  `behavior_inputs.json`.
- `scripts/generate_lcb_behavior_tests.py` runs those prompts through a local
  Transformers model with resumable JSONL output and writes evaluator-ready
  `behavior_inputs.json`.

This keeps the verifier honest: generated tests provide inputs only. Candidate
outputs are compared by execution agreement, and hidden tests are used only for
final measurement.

## Generalization Guardrail

The project now records a cross-benchmark generalization scorecard under
`benchmarks/generalization_scorecard_2026_05_23/`.

Current gate status: `PASS`.

- Full LiveCodeBench `release_v6`: `297/1055` greedy to `403/1055` with `n=8`
  public-test selection, `+0.1005` pass@1.
- LiveCodeBench difficulty splits all improved: easy `+0.1677`, medium
  `+0.1097`, hard `+0.0286`.
- LiveCodeBench platform splits all improved: AtCoder `+0.0880`, Codeforces
  `+0.3333`, LeetCode `+0.1126`.
- EvalPlus holdouts also improved relative to greedy: HumanEval+ `+0.079`,
  MBPP+ `+0.095`.

This is the key anti-overfitting rule for the next stage: a future experiment
cannot be treated as a real improvement if it lifts LiveCodeBench while failing
EvalPlus or regressing important LiveCodeBench slices. The next generated-test,
repair, or post-training run should update this scorecard before being
presented as progress.

## Next Experiments

The S* lite scaling run has now completed for `n=8`. It deliberately separated
public-side construction from hidden-test measurement and used a chunked hidden
replay wrapper so pathological generated programs cannot abort the full suite.
The current headline is `403/1055` (`0.3820`) on full `release_v6`, with one
single-problem worker fatal conservatively counted as failed.

The next active line should keep the same separation:

1. Use the `n=8` public-selection failure set to target public-failing and
   public-pass-hidden-fail candidates separately.
2. Add a repair/generation pass for public-failing candidates before spending
   budget on blind `n=16`; public tests accepted only `618/1055` selected
   candidates, so there is still large public-side headroom.
3. If the repair pass improves both public oracle and held-out hidden replay,
   extend to `n=16` with partial resume.
4. Keep expected-output verifier selection out of the headline until a verifier
   calibration set proves it is better than public-test selection.
5. Update the EvalPlus + LiveCodeBench generalization scorecard before claiming
   a real improvement.

The generation script now supports partial resume for this path:

```bash
python scripts/run_lcb_subset_benchmark.py \
  --lcb-repo /path/to/LiveCodeBench \
  --parquet data/raw/livecodebench/full_release_v6/release_v6_test_prompt_public_only.parquet \
  --output-dir benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n8_full_generate_only \
  --model /path/to/Qwen2.5-Coder-7B-Instruct \
  --n-samples 8 \
  --sample-batch-size 2 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 2048 \
  --generate-only \
  --resume \
  --resume-from-generations benchmarks/livecodebench_full_release_v6_2026_05_22/qwen25_coder_7b_temp08_n4_full_generate_only/generations.json \
  --allow-partial-resume
```

With `--resume-from-generations`, the script seeds the new `n=8` output
directory from the saved `n=4` pool, keeps the first four samples, and generates
only the missing four samples per task. If the new output directory already has
its own `generations.json`, that file takes precedence so interrupted `n=8`
runs can continue normally.
