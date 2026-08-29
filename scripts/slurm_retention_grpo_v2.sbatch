#!/usr/bin/env bash
#SBATCH -J retention-grpo-v2
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH -o /ssd/scxi253/l20-codeforge-retention-rlvr-v2-20260830/logs/grpo-%j.out
#SBATCH -e /ssd/scxi253/l20-codeforge-retention-rlvr-v2-20260830/logs/grpo-%j.err

set -euo pipefail

readonly PROJECT="${PROJECT:-/ssd/scxi253/l20-codeforge-retention-v2}"
readonly RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/l20-codeforge-retention-rlvr-v2-20260830}"
readonly PYTHON="${PYTHON:-/ssd/scxi253/l20-codeforge-gpu-campaign/.venv/bin/python}"
readonly MODEL="${MODEL:-/ssd/scxi253/l20-codeforge-gpu-campaign/hf-home/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242}"
: "${TRAIN_JSONL:?set TRAIN_JSONL}"
: "${REPLAY_JSONL:?set REPLAY_JSONL}"
: "${SFT_ADAPTER:?set SFT_ADAPTER}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tmp/${SLURM_JOB_ID}" "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}/src"
export HF_HOME="/ssd/scxi253/l20-codeforge-gpu-campaign/hf-home"
export TMPDIR="${RUN_ROOT}/tmp/${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m l20_codeforge train-code-grpo \
  "${MODEL}" "${TRAIN_JSONL}" "${OUTPUT_DIR}" \
  --adapter-path "${SFT_ADAPTER}" \
  --replay-jsonl "${REPLAY_JSONL}" \
  --replay-loss-weight "${REPLAY_LOSS_WEIGHT:-0.01}" \
  --replay-max-length "${REPLAY_MAX_LENGTH:-3072}" \
  --max-steps "${MAX_STEPS:-100}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH:-1024}" \
  --limit "${LIMIT:-400}" \
  --learning-rate "${LEARNING_RATE:-0.000001}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
  --gradient-accumulation-steps 4 \
  --num-generations 4 \
  --temperature 0.7 \
  --top-p 0.95 \
  --beta "${BETA:-0.01}" \
  --loss-type dr_grpo \
  --reward-type dense \
  --timeout-seconds 2.0 \
  --execution-workers 6 \
  --load-in-4bit \
  --seed "${SEED:-42}"
