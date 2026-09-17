#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python - <<'PY'
import shutil
import torch

if shutil.which("nvcc") is None:
    raise SystemExit("nvcc was not found. Install a CUDA toolkit matching PyTorch.")
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))
PY

export SELECTIVE_SCAN_MODES="${SELECTIVE_SCAN_MODES:-core}"
export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install -v --no-build-isolation "$REPO_ROOT/kernels/selective_scan"

python - <<'PY'
import torch
import selective_scan_cuda_core
print("selective_scan_cuda_core import: OK")
PY
