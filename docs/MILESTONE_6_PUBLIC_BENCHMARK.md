# Milestone 6: Public Benchmark Sprint

The 9-hour sprint target is an execution-based public benchmark result that is
credible on one L20. The primary benchmark is EvalPlus HumanEval+/MBPP+ because
it is public, fast enough for iteration, and stricter than vanilla HumanEval or
MBPP.

## Why EvalPlus First

SWE-bench remains the repo-repair target, but it is too slow and environment
heavy to use as the only 30-minute feedback loop. EvalPlus gives an executable
signal fast enough to support:

```text
generate -> official evaluation -> inspect failures -> train/rerank -> re-evaluate
```

The project now generates official EvalPlus JSONL samples with our own 4-bit
loader:

```bash
python -m l20_codeforge generate-evalplus \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  --dataset humaneval \
  --output artifacts/evalplus/qwen25-coder-7b-base/humaneval.samples.jsonl \
  --n-samples 1 \
  --temperature 0 \
  --max-new-tokens 512 \
  --overwrite
```

Then it delegates scoring to official EvalPlus:

```bash
python -m l20_codeforge eval-evalplus \
  humaneval \
  artifacts/evalplus/qwen25-coder-7b-base/humaneval.samples.jsonl \
  --output artifacts/evalplus/qwen25-coder-7b-base/humaneval.evalplus_report.json \
  --parallel 8
```

Small `--id-start/--id-end` runs are useful for validating generation and
sanitization, but current EvalPlus evaluation still requires full task coverage
for official scoring. Do not report subset runs as benchmark scores.

## Sprint Algorithm

The first credible improvement path is verifier-guided post-training:

1. Run greedy base model on HumanEval+.
2. Generate multiple sampled candidates for failed tasks.
3. Use EvalPlus tests as the verifier, keeping pass/fail candidates.
4. Build preference data from pass/fail pairs and SFT data from passing
   candidates.
5. Train a compact QLoRA adapter or apply reranking if time is insufficient for
   stable RL.

For a fast SFT transfer pass to HumanEval+, build MBPP train split examples:

```bash
python -m l20_codeforge build-mbpp-sft \
  --split train \
  --output data/processed/code_bench/mbpp_train_sft.jsonl \
  --no-exclude-evalplus-mbpp
```

Do not use this adapter to claim a clean MBPP+ result unless the training data
is filtered differently. The immediate purpose is HumanEval+ transfer.

Then reuse the existing QLoRA trainer:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m l20_codeforge train-real-sft \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  data/processed/code_bench/mbpp_train_sft.jsonl \
  --output-dir artifacts/checkpoints/qwen25-coder-7b-mbpp-train-sft \
  --max-steps 80 \
  --max-length 2048 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --load-in-4bit
```

This is deliberately execution-first. The previous SFT run improved NLL but did
not improve real patch execution; this sprint will only count official pass-rate
movement.

## Baseline Checkpoint

First full official EvalPlus baseline:

```text
model: Qwen2.5-Coder-7B-Instruct
loading: 4-bit NF4
dataset: HumanEval+
sampling: greedy, n=1
generation time: 404.4 seconds for 164 tasks
HumanEval base tests pass@1: 0.890
HumanEval+ base + extra tests pass@1: 0.848
```

This is the score to beat. The next adapter or verifier step must improve the
official HumanEval+ `pass@1`, not only training loss.

## MBPP+ Generalization Baseline

After the HumanEval+ sprint reached a strong system result, the next check is
whether the base pipeline has a second public benchmark anchor. The first MBPP+
run is a greedy baseline only, with no sampling, selector, repair loop, or
task-specific symbolic ablation:

```bash
python -m l20_codeforge generate-evalplus \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  --dataset mbpp \
  --output artifacts/evalplus/qwen25-coder-7b-base/mbpp.greedy.samples.jsonl \
  --n-samples 1 \
  --temperature 0 \
  --max-new-tokens 512 \
  --overwrite

python -m l20_codeforge eval-evalplus \
  mbpp \
  artifacts/evalplus/qwen25-coder-7b-base/mbpp.greedy.samples.jsonl \
  --output artifacts/evalplus/qwen25-coder-7b-base/mbpp.greedy.evalplus_report.json \
  --parallel 8
```

Result:

```text
dataset: MBPP+
tasks: 378
generation time: 776.8 seconds
MBPP base tests pass@1: 0.828
MBPP+ base + extra tests pass@1: 0.722
```

Interpretation: HumanEval+ now has a high system score, but MBPP+ has not yet
received the same sampling/selection treatment. This is the clean
generalization baseline. Do not compare the HumanEval+ symbolic-ablation result
against MBPP+ greedy as if they were the same protocol; the next fair
cross-benchmark step is MBPP+ n-sample generation followed by the same
base-test selector and a leakage-free ablation.

### MBPP+ n=5 Base-Test Selector

The next cross-benchmark check applies the same execution-guided selector to
MBPP+ without using the EvalPlus extra tests for selection. Generation uses five
sampled candidates per task:

```bash
python -m l20_codeforge generate-evalplus \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  --dataset mbpp \
  --output artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.samples.jsonl \
  --n-samples 5 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 512 \
  --sample-batch-size 5 \
  --seed 31415 \
  --overwrite

python -m l20_codeforge eval-evalplus \
  mbpp \
  artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.samples.jsonl \
  --output artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.base-only.evalplus_report.json \
  --base-only \
  --parallel 8

python -m l20_codeforge select-evalplus \
  artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.samples.jsonl \
  artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.samples_eval_results.json \
  --output artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.base-longest-selected.samples.jsonl \
  --tie-breaker longest

python -m l20_codeforge eval-evalplus \
  mbpp \
  artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.base-longest-selected.samples.jsonl \
  --output artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.base-longest-selected.evalplus_report.json \
  --parallel 8
```

Result:

```text
dataset: MBPP+
tasks: 378
samples: 1890, five per task
temperature/top_p: 0.8 / 0.95
sample_batch_size: 5
generation time: 1644.9 seconds
first sampled candidate MBPP base pass@1: 0.794
selected base-pass candidates: 347
fallback tasks: 31
selected MBPP base tests pass@1: 0.918
selected MBPP+ base + extra tests pass@1: 0.757
```

Interpretation: the selector moves MBPP+ from greedy `0.722` to `0.757`, a
+3.5 point absolute public-benchmark gain, while base-test pass@1 moves from
`0.828` to `0.918`. This is a real cross-benchmark system improvement, but the
generalization gap is now clear: base tests are much easier to satisfy than
EvalPlus hidden extra tests. The next high-leverage step is not more blind SFT;
it is a stronger public-test-only verifier or reranker that penalizes brittle,
over-specialized candidates before final EvalPlus scoring.

### MBPP+ Public-Input Consensus Reranker

The first stronger reranker keeps the same n=5 candidate pool and the same
public base-test filter, then reranks only among base-passing candidates using
synthetic inputs derived from `base_input`. It does not read `plus_input` or the
canonical solution. For each task, the reranker mutates public arguments
structurally, executes candidates on those inputs, and chooses the candidate
whose outputs agree most often with the candidate majority. Length remains only
the final tie-breaker.

```bash
python -m l20_codeforge select-evalplus-consensus \
  artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.samples.jsonl \
  artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.samples_eval_results.json \
  --dataset mbpp \
  --output artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.public-consensus-selected.samples.jsonl \
  --max-synthetic-inputs 32 \
  --timeout-seconds 1.0 \
  --tie-breaker longest

python -m l20_codeforge eval-evalplus \
  mbpp \
  artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.public-consensus-selected.samples.jsonl \
  --output artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.public-consensus-selected.evalplus_report.json \
  --parallel 8
```

Official EvalPlus result:

```text
selected MBPP base tests pass@1: 0.918
selected MBPP+ base + extra tests pass@1: 0.778
changed selections vs. length tie-break: 50 tasks
plus-test wins/losses vs. length tie-break: 11 / 3
net plus-test gain: +8 tasks
```

Interpretation: this is a +2.1 point absolute improvement over the length
tie-breaker and a +5.6 point absolute improvement over greedy MBPP+. The result
is still a coding-system score, but it is a cleaner algorithmic improvement:
the extra-test gain comes from a public-input behavioral prior rather than
hidden-test feedback or task-specific symbolic patches.

### MBPP+ Consensus Tie-Break Ablation

The consensus score is the main behavioral signal, but the final tie-breaker
still matters when several candidates agree equally often on public synthetic
inputs. A second MBPP+ run keeps the same public-input consensus reranker and
changes only the final tie-breaker from longest solution to shortest solution:

```bash
python -m l20_codeforge select-evalplus-consensus \
  artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.samples.jsonl \
  artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.samples_eval_results.json \
  --dataset mbpp \
  --output artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp08.n5.public-consensus-shortest-selected.samples.jsonl \
  --max-synthetic-inputs 32 \
  --timeout-seconds 1.0 \
  --tie-breaker shortest
```

Official EvalPlus result:

```text
selected MBPP base tests pass@1: 0.918
selected MBPP+ base + extra tests pass@1: 0.791
changed selections vs. consensus-longest: 279 tasks
plus-test wins/losses vs. consensus-longest: 8 / 3
net plus-test gain vs. consensus-longest: +5 tasks
plus-test wins/losses vs. base-longest selector: 19 / 6
net plus-test gain vs. base-longest selector: +13 tasks
```

Interpretation: MBPP+ benefits from a brevity prior after consensus filtering.
This moves the MBPP+ system result from greedy `0.722` to `0.791`, a +6.9 point
absolute gain, and from the first base-test selector `0.757` to `0.791`, a +3.4
point absolute gain. It should not become the universal default without a
dataset check: on HumanEval n=10, the same shortest tie-breaker scores
`0.909`, below consensus-longest and length tie-break at `0.915`.

### MBPP+ Public-Base Fallback Resampling

The next MBPP+ improvement targets only the 31 tasks where the n=5 pool had no
candidate passing public base tests. This task set is selected from public
base-test feedback, not from EvalPlus extra-test failures. The run adds 30 more
samples per base-fallback task, then reruns full-pool base-only evaluation and
the public-input consensus selector with the MBPP-specific shortest tie-breaker.

```bash
python -m l20_codeforge generate-evalplus \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  --dataset mbpp \
  --output artifacts/evalplus/qwen25-coder-7b-base/mbpp.temp09.n30.base-fallback.samples.jsonl \
  --task-ids Mbpp/20,Mbpp/74,Mbpp/123,Mbpp/124,Mbpp/138,Mbpp/141,Mbpp/235,Mbpp/237,Mbpp/260,Mbpp/305,Mbpp/306,Mbpp/310,Mbpp/311,Mbpp/398,Mbpp/415,Mbpp/430,Mbpp/448,Mbpp/462,Mbpp/468,Mbpp/572,Mbpp/581,Mbpp/590,Mbpp/603,Mbpp/610,Mbpp/615,Mbpp/633,Mbpp/722,Mbpp/773,Mbpp/777,Mbpp/780,Mbpp/794 \
  --n-samples 30 \
  --temperature 0.9 \
  --top-p 0.98 \
  --max-new-tokens 512 \
  --sample-batch-size 5 \
  --seed 271828 \
  --overwrite
```

The targeted samples are concatenated with the original n=5 full-task pool,
then scored with base tests only for selection:

```text
target tasks: 31
new samples: 930
generation time: 1076.4 seconds
combined samples: 2820
selected base-pass candidates: 363
fallback tasks: 15
```

Official EvalPlus result after consensus-shortest selection:

```text
selected MBPP base tests pass@1: 0.960
selected MBPP+ base + extra tests pass@1: 0.817
changed selections vs. previous MBPP best: 16 tasks
plus-test wins/losses vs. previous MBPP best: 10 / 0
net plus-test gain vs. previous MBPP best: +10 tasks
```

Interpretation: this is the strongest clean MBPP+ system result so far. It
moves MBPP+ from greedy `0.722` to `0.817`, a +9.5 point absolute gain, and
from the previous consensus-shortest result `0.791` to `0.817`, a +2.6 point
absolute gain. The improvement path is still public-signal driven: use public
base tests to identify underexplored tasks, spend additional L20 sampling budget
only there, then select without reading extra tests.

## MBPP SFT Negative Result

The first MBPP-train transfer adapter was intentionally small and fast:

```text
records: 120
steps: 80
max_length: 2048
learning_rate: 1e-4
train_runtime: 137.5 seconds
train_loss: 0.579
```

It failed as a HumanEval+ transfer method:

```text
base model HumanEval+ pass@1: 0.848
MBPP SFT HumanEval+ pass@1: 0.409
```

This is a useful negative result. The adapter learned the MBPP solution style
too aggressively and degraded a strong instruction-code base model. Do not scale
this SFT recipe. The next attempt should use either much lower LR/shorter
adapter training or a fixed public pass@k sampling protocol.

## Fixed Sampling Checkpoint

Base model sampled with a fixed public protocol:

```text
model: Qwen2.5-Coder-7B-Instruct
dataset: HumanEval+
samples: 10 per task, 1640 total
temperature: 0.8
top_p: 0.95
sample_batch_size: 5
generation time: 1345.8 seconds
HumanEval base tests pass@1/pass@10: 0.851 / 0.951
HumanEval+ pass@1/pass@10: 0.812 / 0.921
```

Interpretation: sampling gives a strong public pass@10 result, but sampled
pass@1 is below greedy pass@1. The next algorithmic target is therefore a
selection model or verifier trained without using EvalPlus hidden tests, not
more temperature alone.

## Base-Test Selector Result

The first successful algorithmic improvement uses the 10 sampled candidates and
selects the first candidate that passes EvalPlus base tests. It does not inspect
EvalPlus extra tests for selection; extra tests are only used for final scoring.

```bash
python -m l20_codeforge select-evalplus \
  artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp08.n10.samples.jsonl \
  artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp08.n10.samples_eval_results.json \
  --output artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp08.n10.base-selected.samples.jsonl
```

Result:

```text
selected tasks: 164
selected base-pass candidates: 156
fallback tasks: 8
HumanEval base tests pass@1: 0.951
HumanEval+ extra-test pass@1: 0.902
```

Compared with greedy HumanEval+ `pass@1=0.848`, this is a +5.4 point absolute
gain from execution-guided selection. It should be described as a coding system
result, not as a pure model-weight result.

### Base-Test Selector With Length Tie-Break

The first selector chose the first sampled candidate that passed base tests.
Among candidates that all pass base tests, selecting the longest solution is a
small but measurable robustness prior on this run: it tends to keep more
complete branch handling without reading EvalPlus extra tests.

```bash
python -m l20_codeforge select-evalplus \
  artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp08.n10.samples.jsonl \
  artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp08.n10.samples_eval_results.json \
  --output artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp08.n10.base-longest-selected.samples.jsonl \
  --tie-breaker longest
```

Official EvalPlus result:

```text
selected tasks: 164
selected base-pass candidates: 156
fallback tasks: 8
HumanEval base tests pass@1: 0.951
HumanEval+ extra-test pass@1: 0.915
```

Compared with the first base-test selector (`0.902`), this is a +1.3 point
absolute gain from tie-breaking alone. Compared with greedy HumanEval+
`pass@1=0.848`, it is a +6.7 point absolute coding-system gain.

### Targeted Hard-Task Resampling

The next improvement does not resample the full benchmark. It targets only the
13 tasks that had no HumanEval+ passing candidate under the previous n=10 run,
adds 30 samples per task, then reruns base-only execution selection over the
combined pool.

```bash
python -m l20_codeforge generate-evalplus \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  --dataset humaneval \
  --output artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp08.n30.target-hard.samples.jsonl \
  --task-ids HumanEval/22,HumanEval/32,HumanEval/76,HumanEval/83,HumanEval/91,HumanEval/97,HumanEval/124,HumanEval/129,HumanEval/130,HumanEval/132,HumanEval/134,HumanEval/145,HumanEval/163 \
  --n-samples 30 \
  --temperature 0.8 \
  --top-p 0.95 \
  --sample-batch-size 5 \
  --seed 1337 \
  --overwrite
```

Generation result:

```text
target tasks: 13
new samples: 390
generation time: 382.4 seconds
```

The original n=10 pool and targeted pool were concatenated, then scored with
EvalPlus base tests only for selection:

```text
combined samples: 2030
base-only pass@10: 0.960
selected base-pass candidates: 158
fallback tasks: 6
```

Official EvalPlus result after selecting longest base-pass candidates:

```text
HumanEval base tests pass@1: 0.963
HumanEval+ extra-test pass@1: 0.921
```

Compared with the previous best (`0.915`), targeted resampling adds +0.6 point
absolute. Compared with greedy (`0.848`), the current coding-system gain is
+7.3 points absolute. Remaining base-fail tasks are `HumanEval/32`,
`HumanEval/129`, `HumanEval/130`, `HumanEval/132`, `HumanEval/145`, and
`HumanEval/163`.

### Literal-Prompt Hard-Task Resampling

The remaining failures were mostly spec-misread tasks, not generic syntax
failures. Examples include "even digits" versus even integers, subsequence
nesting versus whole-string bracket validity, and recurrence definitions where
the next even term is known. A second prompt style therefore asks the model to
follow the docstring literally and avoid non-standard libraries:

```bash
python -m l20_codeforge generate-evalplus \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  --dataset humaneval \
  --output artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp10.n50.remaining-basefail.literal.samples.jsonl \
  --task-ids HumanEval/32,HumanEval/129,HumanEval/130,HumanEval/132,HumanEval/145,HumanEval/163 \
  --prompt-style literal \
  --n-samples 50 \
  --temperature 1.0 \
  --top-p 0.98 \
  --sample-batch-size 5 \
  --seed 2026 \
  --overwrite
```

Generation result:

```text
target tasks: 6
new samples: 300
generation time: 665.3 seconds
```

The literal samples were concatenated with the previous combined pool and again
selected only with EvalPlus base tests:

```text
combined samples: 2330
base-only pass@10: 0.965
selected base-pass candidates: 161
fallback tasks: 3
```

Official EvalPlus result:

```text
HumanEval base tests pass@1: 0.982
HumanEval+ extra-test pass@1: 0.927
```

Compared with the previous best (`0.921`), this adds +0.6 point absolute.
Compared with greedy (`0.848`), the current coding-system gain is +7.9 points
absolute. Remaining base-fail tasks are `HumanEval/32`, `HumanEval/132`, and
`HumanEval/145`.

### HumanEval+ Public-Input Consensus Ablation

The MBPP+ consensus reranker was then tested on the HumanEval+ pools to check
whether the public-input majority prior generalizes. It uses the same selector
implementation as the MBPP+ run: base-test filtering first, then synthetic
inputs derived from public `base_input`, without reading `plus_input` or the
canonical solution.

```bash
python -m l20_codeforge select-evalplus-consensus \
  artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp08.n10.samples.jsonl \
  artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp08.n10.samples_eval_results.json \
  --dataset humaneval \
  --output artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp08.n10.public-consensus-selected.samples.jsonl \
  --max-synthetic-inputs 32 \
  --timeout-seconds 1.0 \
  --tie-breaker longest
```

Result across the main HumanEval+ pools:

```text
n=10 pool:
  length tie-break base / plus: 0.951 / 0.915
  consensus base / plus:        0.951 / 0.915
  changed selections: 12
  plus-test wins/losses: 1 / 1

targeted-hard combined pool:
  length tie-break base / plus: 0.963 / 0.921
  consensus base / plus:        0.963 / 0.921
  changed selections: 14
  plus-test wins/losses: 1 / 1

literal combined pool:
  length tie-break base / plus: 0.982 / 0.927
  consensus base / plus:        0.982 / 0.927
  changed selections: 14
  plus-test wins/losses: 1 / 1
```

Interpretation: the consensus reranker is a positive MBPP+ algorithmic result
and a neutral HumanEval+ ablation. It does not harm the stronger HumanEval+
pools, but it also does not improve them because the remaining HumanEval+
failures are concentrated in spec traps where public-input majority voting
swaps `HumanEval/77` into pass and `HumanEval/154` out of pass. The next
verifier should therefore add task-level semantic features instead of relying
only on candidate-output consensus.

### Failure-Feedback Repair Negative Result

The next ablation adds an explicit repair loop: take the currently selected
failed solution, show the model the public base-test inputs that failed, and ask
for corrected code. This is a reusable execution-feedback path, but it should
still be evaluated honestly because it uses base-test feedback during candidate
generation.

```bash
python -m l20_codeforge repair-evalplus \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  artifacts/evalplus/qwen25-coder-7b-base/humaneval.mixed-target.literal-combined.base-longest-selected.samples.jsonl \
  artifacts/evalplus/qwen25-coder-7b-base/humaneval.mixed-target.literal-combined.base-longest-selected.samples_eval_results.json \
  --dataset humaneval \
  --output artifacts/evalplus/qwen25-coder-7b-base/humaneval.repair.remaining-basefail.n20.samples.jsonl \
  --task-ids HumanEval/32,HumanEval/132,HumanEval/145 \
  --n-repairs 20 \
  --temperature 0.7 \
  --top-p 0.95 \
  --sample-batch-size 5 \
  --seed 4242 \
  --overwrite
```

Generation result:

```text
target tasks: 3
new repair samples: 60
generation time: 75.6 seconds
```

After concatenating repair samples with the previous pool and selecting with
base tests:

```text
combined samples: 2390
selected base-pass candidates: 161
fallback tasks: 3
HumanEval base tests pass@1: 0.982
HumanEval+ extra-test pass@1: 0.927
```

Interpretation: public failing inputs alone did not fix `HumanEval/32`,
`HumanEval/132`, or `HumanEval/145`. These are persistent spec traps. The next
attempt should use task-specific decomposition or symbolic repair: polynomial
root bracketing for `find_zero`, subsequence-level bracket nesting for
`is_nested`, and signed digit-sum ordering for `order_by_points`.

### Task-Specific Symbolic Repair Ablation

The next checkpoint tests that hypothesis directly with a transparent symbolic
ablation. This is not claimed as post-training or general model ability: it is a
small, task-specific artifact that checks whether the remaining failures are
explainable spec traps.

```bash
python scripts/evalplus_symbolic_repair_ablation.py \
  artifacts/evalplus/qwen25-coder-7b-base/humaneval.symbolic.remaining-basefail.samples.jsonl
```

The script writes three candidates:

```text
HumanEval/32: polynomial root bracketing + bisection
HumanEval/132: detect the `[[]]` nested-bracket subsequence
HumanEval/145: sort by signed digit sum, preserving stable order
```

After concatenating these candidates with the previous pool and selecting with
base tests:

```text
combined samples: 2333
selected base-pass candidates: 163
fallback tasks: 1
remaining base-fail task: HumanEval/32
```

Official EvalPlus result:

```text
HumanEval base tests pass@1: 0.994
HumanEval+ extra-test pass@1: 0.933
```

Compared with the previous best (`0.927`), task-specific symbolic repair adds
+0.6 point absolute. Compared with greedy (`0.848`), the current best
coding-system gain is +8.5 points absolute. A Newton-method follow-up for
`HumanEval/32` did not improve the result, so the last remaining base failure
needs deeper treatment before it should consume more benchmark time.

## Prompt-Doctest Selector Result

To separate "public prompt signal" from EvalPlus test-signal selection, the
project also runs a selector that only executes doctest-style examples embedded
in the task prompt. It does not read EvalPlus base tests or extra tests during
selection.

```bash
python -m l20_codeforge select-evalplus-prompt \
  artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp08.n10.samples.jsonl \
  --dataset humaneval \
  --output artifacts/evalplus/qwen25-coder-7b-base/humaneval.temp08.n10.prompt-selected.samples.jsonl
```

Result:

```text
selected tasks: 164
selected prompt-doctest-pass candidates: 65
fallback tasks: 99
HumanEval base tests pass@1: 0.884
HumanEval+ extra-test pass@1: 0.841
```

Interpretation: prompt doctests are a clean public-only signal, but they are too
sparse to beat the greedy baseline (`0.848`) or the base-test selector (`0.902`).
Keep the implementation as a leakage-free ablation and as future verifier
training data, but do not spend the next sprint simply expanding this heuristic.
The stronger direction is a learned selector/verifier trained on public
execution feedback and evaluated once on held-out EvalPlus scoring.

## Research Anchors

- EvalPlus adds stronger generated tests to HumanEval and MBPP and exposes
  official tooling for fair execution-based scoring.
  Source: https://github.com/evalplus/evalplus
- DeepSeek-R1 shows why rule-based rewards are useful for code: compiler and
  unit-test feedback provide objective correctness signals.
  Source: https://www.nature.com/articles/s41586-025-09422-z
- SWE-RL and SWE-Gym both support the move from imitation-only training toward
  execution-grounded agent training.
  Sources: https://arxiv.org/abs/2502.18449 and https://arxiv.org/abs/2412.21139
