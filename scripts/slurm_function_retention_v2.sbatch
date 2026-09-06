#!/usr/bin/env bash
#SBATCH -J function-retention-v2
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH -o /ssd/scxi253/l20-codeforge-retention-rlvr-v2-20260830/logs/function-eval-%j.out
#SBATCH -e /ssd/scxi253/l20-codeforge-retention-rlvr-v2-20260830/logs/function-eval-%j.err

set -euo pipefail

readonly PROJECT="${PROJECT:-/ssd/scxi253/l20-codeforge-retention-v2}"
readonly RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/l20-codeforge-retention-rlvr-v2-20260830}"
readonly PYTHON="${PYTHON:-/ssd/scxi253/l20-codeforge-gpu-campaign/.venv/bin/python}"
readonly MODEL="${MODEL:-/ssd/scxi253/l20-codeforge-gpu-campaign/hf-home/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242}"
: "${TASKS_JSONL:?set TASKS_JSONL}"
: "${OUTPUT:?set OUTPUT}"

readonly ADAPTER_PATH="${ADAPTER_PATH:-}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tmp/${SLURM_JOB_ID}" "$(dirname "${OUTPUT}")"
export PYTHONPATH="${PROJECT}/src"
export HF_HOME="/ssd/scxi253/l20-codeforge-gpu-campaign/hf-home"
export TMPDIR="${RUN_ROOT}/tmp/${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

command=(
  "${PYTHON}" -m l20_codeforge generate-function-retention-rollouts
  "${MODEL}" "${TASKS_JSONL}" "${OUTPUT}"
  --n-samples 1
  --temperature 0.0
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}"
  --batch-size "${BATCH_SIZE:-2}"
  --timeout-seconds "${TIMEOUT_SECONDS:-4.0}"
  --seed "${SEED:-42}"
  --overwrite
)
if [[ -n "${ADAPTER_PATH}" ]]; then
  command+=(--adapter-path "${ADAPTER_PATH}")
fi
"${command[@]}"

sha256sum "${TASKS_JSONL}" "${OUTPUT}" >"${OUTPUT}.sha256sums.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
  >"${OUTPUT}.gpu_receipt.csv"
