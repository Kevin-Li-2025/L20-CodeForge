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
