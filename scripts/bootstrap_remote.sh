#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -d ".venv" ]; then
  uv venv --python 3.12 --seed --no-managed-python .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip

# Match the host driver-reported CUDA 12.4 runtime. Keep torch separate from the
# project dependencies so vLLM or future training stacks can be isolated.
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

uv pip install -e ".[train,agent,dev]"

python scripts/check_gpu.py
python -m l20_codeforge profile
pytest -q
