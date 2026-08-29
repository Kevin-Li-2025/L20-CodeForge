#!/usr/bin/env bash
#SBATCH -J evalplus-greedy
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH -o /ssd/scxi253/l20-codeforge-base-sft-rlvr-20260829/logs/evalplus-%A_%a.out
#SBATCH -e /ssd/scxi253/l20-codeforge-base-sft-rlvr-20260829/logs/evalplus-%A_%a.err

set -euo pipefail

readonly PROJECT="${PROJECT:-/ssd/scxi253/l20-codeforge-verifier-phase-a}"
readonly RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/l20-codeforge-base-sft-rlvr-20260829}"
readonly PYTHON="${PYTHON:-/ssd/scxi253/l20-codeforge-gpu-campaign/.venv/bin/python}"
readonly MODEL="${MODEL:-/ssd/scxi253/l20-codeforge-gpu-campaign/hf-home/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242}"
: "${DATASET:?set DATASET to humaneval or mbpp}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

readonly SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
readonly SHARD_COUNT="${SHARD_COUNT:-4}"
readonly ADAPTER_PATH="${ADAPTER_PATH:-}"
if [[ "${DATASET}" != "humaneval" && "${DATASET}" != "mbpp" ]]; then
  echo "unsupported DATASET=${DATASET}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tmp/${SLURM_JOB_ID}" "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT}/src"
export HF_HOME="/ssd/scxi253/l20-codeforge-gpu-campaign/hf-home"
export TMPDIR="${RUN_ROOT}/tmp/${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

command=(
  "${PYTHON}" -m l20_codeforge generate-evalplus
  "${MODEL}"
  --dataset "${DATASET}"
  --output "${OUTPUT_DIR}/${DATASET}-shard-${SHARD_INDEX}.samples.jsonl"
  --n-samples 1
  --shard-index "${SHARD_INDEX}"
  --shard-count "${SHARD_COUNT}"
  --temperature 0.0
  --max-new-tokens "${MAX_NEW_TOKENS:-512}"
  --sample-batch-size 1
  --seed 42
)
if [[ -n "${ADAPTER_PATH}" ]]; then
  command+=(--adapter-path "${ADAPTER_PATH}")
fi
"${command[@]}"

sha256sum "${OUTPUT_DIR}/${DATASET}-shard-${SHARD_INDEX}.samples.jsonl" \
  >"${OUTPUT_DIR}/${DATASET}-shard-${SHARD_INDEX}.sha256sum.txt"
