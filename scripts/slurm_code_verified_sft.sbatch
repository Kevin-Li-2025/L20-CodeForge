#!/usr/bin/env bash
#SBATCH -J code-verified-sft
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH -o /ssd/scxi253/l20-codeforge-base-sft-rlvr-20260829/logs/sft-%j.out
#SBATCH -e /ssd/scxi253/l20-codeforge-base-sft-rlvr-20260829/logs/sft-%j.err

set -euo pipefail

readonly PROJECT="${PROJECT:-/ssd/scxi253/l20-codeforge-verifier-phase-a}"
readonly RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/l20-codeforge-base-sft-rlvr-20260829}"
readonly PYTHON="${PYTHON:-/ssd/scxi253/l20-codeforge-gpu-campaign/.venv/bin/python}"
readonly MODEL="${MODEL:-/ssd/scxi253/l20-codeforge-gpu-campaign/hf-home/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242}"
: "${TRAIN_JSONL:?set TRAIN_JSONL}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tmp/${SLURM_JOB_ID}" "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}/src"
export HF_HOME="/ssd/scxi253/l20-codeforge-gpu-campaign/hf-home"
export TMPDIR="${RUN_ROOT}/tmp/${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m l20_codeforge train-real-sft \
  "${MODEL}" "${TRAIN_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-steps "${MAX_STEPS:-100}" \
  --max-length "${MAX_LENGTH:-3072}" \
  --limit "${LIMIT:-400}" \
  --learning-rate "${LEARNING_RATE:-0.0001}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --completion-only-loss \
  --load-in-4bit \
  --seed 42
