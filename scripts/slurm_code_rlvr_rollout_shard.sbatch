#!/usr/bin/env bash
#SBATCH -J code-rollout-shard
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH -o /ssd/scxi253/l20-codeforge-base-sft-rlvr-20260829/logs/rollout-shard-%A_%a.out
#SBATCH -e /ssd/scxi253/l20-codeforge-base-sft-rlvr-20260829/logs/rollout-shard-%A_%a.err

set -euo pipefail

readonly PROJECT="${PROJECT:-/ssd/scxi253/l20-codeforge-verifier-phase-a}"
readonly RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/l20-codeforge-base-sft-rlvr-20260829}"
readonly PYTHON="${PYTHON:-/ssd/scxi253/l20-codeforge-gpu-campaign/.venv/bin/python}"
readonly MODEL="${MODEL:-/ssd/scxi253/l20-codeforge-gpu-campaign/hf-home/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242}"
: "${TASKS_JSONL:?set TASKS_JSONL}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

readonly SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
readonly SHARD_COUNT="${SHARD_COUNT:-4}"
readonly N_SAMPLES="${N_SAMPLES:-1}"
readonly TEMPERATURE="${TEMPERATURE:-0.0}"
readonly TOP_P="${TOP_P:-0.95}"
readonly MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
readonly BATCH_SIZE="${BATCH_SIZE:-2}"
readonly ADAPTER_PATH="${ADAPTER_PATH:-}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tmp/${SLURM_JOB_ID}" "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}/src"
export HF_HOME="/ssd/scxi253/l20-codeforge-gpu-campaign/hf-home"
export TMPDIR="${RUN_ROOT}/tmp/${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

command=(
  "${PYTHON}" -m l20_codeforge generate-code-rollouts
  "${MODEL}" "${TASKS_JSONL}" "${OUTPUT_DIR}/shard-${SHARD_INDEX}.jsonl"
  --n-samples "${N_SAMPLES}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --batch-size "${BATCH_SIZE}"
  --execution-workers "${EXECUTION_WORKERS:-6}"
  --seed 42
  --shard-index "${SHARD_INDEX}"
  --shard-count "${SHARD_COUNT}"
)
if [[ -n "${ADAPTER_PATH}" ]]; then
  command+=(--adapter-path "${ADAPTER_PATH}")
fi
"${command[@]}"

sha256sum "${TASKS_JSONL}" "${OUTPUT_DIR}/shard-${SHARD_INDEX}.jsonl" \
  >"${OUTPUT_DIR}/shard-${SHARD_INDEX}.sha256sums.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
  >"${OUTPUT_DIR}/shard-${SHARD_INDEX}.gpu_receipt.csv"
