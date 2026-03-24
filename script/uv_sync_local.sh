#!/usr/bin/env bash
# Local `uv sync` when /usr/local/cuda -> 11.8 but lock uses PyTorch cu121 + nvidia-curobo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9;9.0+PTX}"
exec uv sync "$@"
