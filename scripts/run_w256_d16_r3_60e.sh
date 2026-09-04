#!/usr/bin/env bash
set -euo pipefail

repo=$(readlink -f "${1:-.}")
py=${PYTHON_BIN:-/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python}
study=.experiments/w256_d16_60e_20260904
config=configs/pose3d/graph_posemamba_h36m_w256_d16_stable_r3_60e.yaml
prefix=runs/w256_d16_r3_60e/D16_w256_d16_stable_r3_seed0
verification=verification/w256_d16_r3_60e
preflight=$study/PRELAUNCH_PASS_R3.json

cd "$repo"
mkdir -p "$study/runtime" "$(dirname "$prefix")" "$verification" launch_logs
exec 9>"$study/r3_queue.lock"
flock -n 9 || { echo "D16 R3 queue lock is held" >&2; exit 3; }
[[ -f "$study/USER_AUTHORIZED.txt" ]] || { echo "authorization missing" >&2; exit 4; }
[[ -f "$preflight" ]] || { echo "R3 preflight missing" >&2; exit 5; }
if ! git diff --quiet -- lib/model/PoseMamba.py lib/model/mambablocks.py lib/utils/learning.py train.py tools/benchmark_training.py "$config"; then
  echo "tracked R3 source is dirty" >&2
  exit 6
fi

"$py" - "$preflight" "$config" "$(git rev-parse HEAD)" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
config_hash = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
if payload.get("status") != "PASS":
    raise RuntimeError("R3 preflight status is not PASS")
if payload.get("source_commit") != sys.argv[3]:
    raise RuntimeError("R3 preflight belongs to a different source commit")
if payload.get("config_sha256") != config_hash:
    raise RuntimeError("R3 preflight belongs to a different config")
PY

active=$(
  for pid in $(pgrep -f '[p]ython.*train\.py' || true); do
    cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
    [[ "$cwd" == "$repo" ]] && ps -p "$pid" -o pid=,args=
  done
)
[[ -z "$active" ]] || { echo "active target-repo training: $active" >&2; exit 7; }
if compgen -G "${prefix}_*" >/dev/null; then echo "existing D16 R3 run prefix" >&2; exit 8; fi

dataset=data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl
expected=73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175
actual=$(sha256sum "$dataset" | awk '{print $1}')
[[ "$actual" == "$expected" ]] || { echo "dataset hash mismatch" >&2; exit 9; }

printf 'source_commit=%s\nstarted=%s\npython=%s\ndataset_sha256=%s\nconfig_sha256=%s\n' \
  "$(git rev-parse HEAD)" "$(date --iso-8601=seconds)" "$py" "$actual" \
  "$(sha256sum "$config" | awk '{print $1}')" > "$study/runtime/R3_identity.txt"
printf 'config=%s\nstarted=%s\n' "$config" "$(date --iso-8601=seconds)" > "$study/runtime/R3_RUNNING.txt"

parent_pid=$$
(
  echo timestamp,phase,gpu_used_mib,gpu_free_mib,gpu_util_percent,temp_c,power_w
  while kill -0 "$parent_pid" 2>/dev/null; do
    values=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits)
    echo "$(date --iso-8601=seconds),D16_R3,$values"
    sleep 5
  done
) > "$study/runtime/R3_gpu_monitor.csv" 2> "$study/runtime/R3_gpu_monitor.stderr" &
monitor_pid=$!
echo "$monitor_pid" > "$study/runtime/R3_gpu_monitor.pid"
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

CUDA_VISIBLE_DEVICES=0 "$py" train.py --config "$config" --checkpoint "$prefix" --seed 0
run_dir=$(find "$(dirname "$prefix")" -maxdepth 1 -type d -name "$(basename "$prefix")_*" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
[[ -n "$run_dir" ]] || { echo "D16 R3 run directory not found" >&2; exit 10; }

"$py" - "$run_dir" <<'PY'
import os
import sys
import torch

run = sys.argv[1]
for name in ("latest_epoch.bin", "latest_ema_epoch.bin", "best_epoch.bin", "best_ema_epoch.bin"):
    path = os.path.join(run, name)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if name.startswith("latest") and int(payload.get("epoch", -1)) != 60:
        raise RuntimeError(f"{path}: expected epoch 60")
    if not payload.get("model_pos"):
        raise RuntimeError(f"{path}: missing model_pos")
PY

for checkpoint_name in best_epoch.bin best_ema_epoch.bin; do
  label=${checkpoint_name%.bin}
  CUDA_VISIBLE_DEVICES=0 "$py" train.py --config "$config" \
    --evaluate "$run_dir/$checkpoint_name" --checkpoint "$verification/$label" --seed 0 \
    > "$run_dir/verification_${label}.stdout" 2>&1
done
printf 'run_dir=%s\ncompleted=%s\n' "$run_dir" "$(date --iso-8601=seconds)" > "$study/runtime/R3_COMPLETE.txt"

