# Milestone 5: L20 7B Real SFT

This milestone proves that L20 CodeForge can run a real 7B coding-model
post-training pass on one NVIDIA L20 using real SWE-bench data.

## Model Staging

The L20 host could not reach `huggingface.co` directly, and Tailscale file
transfer from the Mac was too slow for a 15 GiB model. The reliable path was to
download from `hf-mirror.com` on the L20 host:

```bash
HF_ENDPOINT=https://hf-mirror.com hf download \
  Qwen/Qwen2.5-Coder-7B-Instruct \
  config.json generation_config.json model.safetensors.index.json \
  model-00001-of-00004.safetensors \
  model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors \
  model-00004-of-00004.safetensors \
  tokenizer.json tokenizer_config.json merges.txt vocab.json \
  --local-dir /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct
```

The staged model is about 15 GiB and loads offline with
`AutoTokenizer`/`AutoConfig`.

## Training Runs

Small 7B QLoRA smoke on SWE-bench Lite:

```bash
python -m l20_codeforge train-real-sft \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  data/processed/real_sft/swe_bench_lite_sft.jsonl \
  --output-dir artifacts/checkpoints/qwen25-coder-7b-real-sft-smoke \
  --max-steps 5 \
  --limit 64 \
  --max-length 2048 \
  --gradient-accumulation-steps 4 \
  --load-in-4bit
```

Result:

```json
{
  "records": 64,
  "max_steps": 5,
  "max_length": 2048,
  "train_runtime": 18.2882,
  "train_loss": 1.45111186504364
}
```

4096-token probe on SWE-bench Verified:

```bash
python -m l20_codeforge train-real-sft \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  data/processed/real_sft/swe_bench_verified_sft.jsonl \
  --output-dir artifacts/checkpoints/qwen25-coder-7b-real-sft-4096-bs2-probe \
  --max-steps 5 \
  --limit 64 \
  --max-length 4096 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --load-in-4bit
```

Result:

```json
{
  "records": 64,
  "max_steps": 5,
  "max_length": 4096,
  "train_runtime": 86.0627,
  "train_loss": 1.1263612270355225
}
```

Full first pass on SWE-bench Verified:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m l20_codeforge train-real-sft \
  /home/hhai/model-cache/Qwen2.5-Coder-7B-Instruct \
  data/processed/real_sft/swe_bench_verified_sft.jsonl \
  --output-dir artifacts/checkpoints/qwen25-coder-7b-verified-sft-4096-epoch1 \
  --max-steps 64 \
  --limit 500 \
  --max-length 4096 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --load-in-4bit
```

Result:

```json
{
  "records": 500,
  "max_steps": 64,
  "max_length": 4096,
  "train_runtime": 1011.2366,
  "train_samples_per_second": 0.506,
  "train_steps_per_second": 0.063,
  "total_flos": 5.897038193799782e+16,
  "train_loss": 0.9146799212321639
}
```

The final adapter is at:

```text
artifacts/checkpoints/qwen25-coder-7b-verified-sft-4096-epoch1/final
```

The full artifact is 331 MiB, and `final/adapter_model.safetensors` is about
81 MiB.

## L20 Boundary

Stable configuration:

```text
model: Qwen2.5-Coder-7B-Instruct
method: QLoRA 4-bit NF4
max_length: 4096
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
effective_batch_size: 8
observed_peak_memory: about 29.6 GiB
```

Failed configuration:

```text
max_length: 4096
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
observed_memory_before_failure: about 41.9 GiB
failure: CUDA OOM while materializing shifted logits
```

The next optimization target is not model loading; it is loss memory. Avoiding
full-logit materialization or using assistant-token-only loss should let the L20
use a larger microbatch or longer examples.

## Verification

The final adapter was loaded with the base model in 4-bit mode and produced a
valid Python `two_sum` implementation from a short coding prompt.
