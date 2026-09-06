#!/usr/bin/env bash
#SBATCH -J evalplus-score
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/l20-codeforge-base-sft-rlvr-20260829/logs/evalplus-score-%j.out
#SBATCH -e /ssd/scxi253/l20-codeforge-base-sft-rlvr-20260829/logs/evalplus-score-%j.err

set -euo pipefail

readonly PROJECT="${PROJECT:-/ssd/scxi253/l20-codeforge-verifier-phase-a}"
readonly RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/l20-codeforge-base-sft-rlvr-20260829}"
readonly PYTHON="${PYTHON:-/ssd/scxi253/l20-codeforge-gpu-campaign/.venv/bin/python}"
: "${DATASET:?set DATASET to humaneval or mbpp}"
: "${SAMPLES:?set SAMPLES to the merged greedy sample JSONL}"
: "${OUTPUT:?set OUTPUT to the official report JSON}"
if [[ "${DATASET}" != "humaneval" && "${DATASET}" != "mbpp" ]]; then
  echo "unsupported DATASET=${DATASET}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tmp/${SLURM_JOB_ID}" "$(dirname "${OUTPUT}")"
export PYTHONPATH="${PROJECT}/src"
export XDG_CACHE_HOME="${RUN_ROOT}/xdg-cache"
export TMPDIR="${RUN_ROOT}/tmp/${SLURM_JOB_ID}"

"${PYTHON}" -m l20_codeforge eval-evalplus \
  "${DATASET}" "${SAMPLES}" \
  --output "${OUTPUT}" \
  --parallel "${SLURM_CPUS_PER_TASK:-6}"

sha256sum "${SAMPLES}" "${OUTPUT}" >"${OUTPUT}.sha256sums.txt"
