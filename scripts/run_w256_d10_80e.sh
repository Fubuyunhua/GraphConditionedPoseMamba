#!/usr/bin/env bash
set -euo pipefail

repo=$(readlink -f "${1:-.}")
py=${PYTHON_BIN:-/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python}
study=.experiments/model_scaling_80e_20260903
config=configs/pose3d/graph_posemamba_h36m_w256_d10_scale_80e.yaml
prefix=runs/model_scaling_80e/S2_w256_d10_seed0
verification=verification/model_scaling_80e

cd "$repo"
mkdir -p "$study/runtime" "$(dirname "$prefix")" "$verification" launch_logs
exec 9>"$study/S2_override_queue.lock"
flock -n 9 || { echo "S2 override queue lock is held" >&2; exit 3; }
[[ -f "$study/S2_OVERRIDE_AUTHORIZED.txt" ]] || { echo "override authorization missing" >&2; exit 4; }
[[ -f "$study/PRELAUNCH_PASS.json" ]] || { echo "preflight missing" >&2; exit 5; }

if ! git diff --quiet -- lib/model/PoseMamba.py lib/model/mambablocks.py \
  lib/utils/learning.py train.py "$config"; then
  echo "tracked S2 source is dirty" >&2
  exit 6
fi
active=$(
  for pid in $(pgrep -f '[p]ython.*train\.py' || true); do
    cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
    [[ "$cwd" == "$repo" ]] && ps -p "$pid" -o pid=,args=
  done
)
[[ -z "$active" ]] || { echo "active target-repo training: $active" >&2; exit 7; }
if compgen -G "${prefix}_*" >/dev/null; then
  echo "refusing existing S2 run prefix" >&2
  exit 8
fi

dataset=data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl
expected=73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175
actual=$(sha256sum "$dataset" | awk '{print $1}')
[[ "$actual" == "$expected" ]] || { echo "dataset hash mismatch" >&2; exit 9; }

printf 'source_commit=%s\nstarted=%s\npython=%s\ndataset_sha256=%s\n' \
  "$(git rev-parse HEAD)" "$(date --iso-8601=seconds)" "$py" "$actual" \
  > "$study/runtime/S2_override_identity.txt"
printf 'id=S2\nconfig=%s\nstarted=%s\nreason=user_prioritized_w256\n' \
  "$config" "$(date --iso-8601=seconds)" > "$study/runtime/S2_RUNNING.txt"

parent_pid=$$
(
  echo timestamp,phase,gpu_used_mib,gpu_free_mib,gpu_util_percent,temp_c,power_w
  while kill -0 "$parent_pid" 2>/dev/null; do
    values=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits)
    echo "$(date --iso-8601=seconds),S2,$values"
    sleep 5
  done
) > "$study/runtime/S2_gpu_monitor.csv" 2> "$study/runtime/S2_gpu_monitor.stderr" &
monitor_pid=$!
echo "$monitor_pid" > "$study/runtime/S2_gpu_monitor.pid"
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

CUDA_VISIBLE_DEVICES=0 "$py" train.py \
  --config "$config" --checkpoint "$prefix" --seed 0

run_dir=$(find "$(dirname "$prefix")" -maxdepth 1 -type d \
  -name "$(basename "$prefix")_*" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
[[ -n "$run_dir" ]] || { echo "S2 run directory not found" >&2; exit 10; }
"$py" - "$run_dir" <<'PY'
import os
import sys
import torch

run = sys.argv[1]
for name in ("latest_epoch.bin", "latest_ema_epoch.bin", "best_epoch.bin", "best_ema_epoch.bin"):
    path = os.path.join(run, name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if name.startswith("latest") and int(checkpoint.get("epoch", -1)) != 80:
        raise RuntimeError(f"{path}: expected epoch 80, got {checkpoint.get('epoch')}")
    if not checkpoint.get("model_pos"):
        raise RuntimeError(f"{path}: missing model_pos")
PY
for checkpoint_name in best_epoch.bin best_ema_epoch.bin; do
  label=${checkpoint_name%.bin}
  CUDA_VISIBLE_DEVICES=0 "$py" train.py \
    --config "$config" --evaluate "$run_dir/$checkpoint_name" \
    --checkpoint "$verification/S2_${label}" --seed 0 \
    > "$run_dir/verification_${label}.stdout" 2>&1
done
printf 'id=S2\nrun_dir=%s\ncompleted=%s\n' "$run_dir" "$(date --iso-8601=seconds)" \
  > "$study/runtime/S2_COMPLETE.txt"
