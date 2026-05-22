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
