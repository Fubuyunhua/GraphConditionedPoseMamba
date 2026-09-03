#!/usr/bin/env bash
set -euo pipefail

repo=$(readlink -f "${1:-.}")
ablation_repo=$(readlink -f "${2:-/scratch/home/caiwei/GraphConditionedPoseMamba_5090_20260902}")
py=${PYTHON_BIN:-/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python}
study=.experiments/model_scaling_80e_20260903
ablation_study=.experiments/minimal_ablation_80e_20260903
root=runs/model_scaling_80e
verification=verification/model_scaling_80e

cd "$repo"
mkdir -p "$study/runtime" "$root" "$verification" launch_logs
exec 9>"$study/queue.lock"
flock -n 9 || { echo "scale queue lock is held" >&2; exit 3; }

[[ -f "$study/USER_AUTHORIZED.txt" ]] || { echo "authorization missing" >&2; exit 4; }
[[ -f "$study/PRELAUNCH_PASS.json" ]] || { echo "preflight missing" >&2; exit 5; }
[[ -f "$ablation_repo/$ablation_study/SEQUENCE_COMPLETE.txt" ]] || {
  echo "A1/A2 sequence is not complete" >&2
  exit 6
}

if ! git diff --quiet -- lib/model/PoseMamba.py lib/model/mambablocks.py \
  lib/utils/learning.py train.py tools/summarize_scale_models.py \
  configs/pose3d/graph_posemamba_h36m_w128_d20_scale_80e.yaml \
  configs/pose3d/graph_posemamba_h36m_w256_d10_scale_80e.yaml; then
  echo "tracked scale source is dirty" >&2
  exit 7
fi

active=$(
  for pid in $(pgrep -f '[p]ython.*train\.py' || true); do
    cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
    if [[ "$cwd" == "$repo" ]]; then
      ps -p "$pid" -o pid=,args=
    fi
  done
)
[[ -z "$active" ]] || { echo "active target-repo training: $active" >&2; exit 8; }

dataset=data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl
expected=73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175
actual=$(sha256sum "$dataset" | awk '{print $1}')
[[ "$actual" == "$expected" ]] || { echo "dataset hash mismatch" >&2; exit 9; }

"$py" - "$study/PRELAUNCH_PASS.json" "$(git rev-parse HEAD)" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
if report.get("status") != "PASS":
    raise RuntimeError("scale preflight status is not PASS")
if report.get("source_commit") != sys.argv[2]:
    raise RuntimeError("scale preflight belongs to a different source commit")
for path, expected in report.get("config_sha256", {}).items():
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"stale scale preflight for {path}")
PY

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "invalid free-memory reading" >&2; exit 10; }
(( free_mib >= 24576 )) || {
  echo "only ${free_mib} MiB GPU memory is free; require 24576" >&2
  exit 11
}

printf 'source_commit=%s\nstarted=%s\npython=%s\ndataset_sha256=%s\n' \
  "$(git rev-parse HEAD)" "$(date --iso-8601=seconds)" "$py" "$actual" \
  > "$study/runtime/queue_identity.txt"

parent_pid=$$
(
  echo timestamp,phase,gpu_used_mib,gpu_free_mib,gpu_util_percent,temp_c,power_w
  while kill -0 "$parent_pid" 2>/dev/null; do
    if [[ -f "$study/runtime/S2_RUNNING.txt" ]]; then phase=S2
    elif [[ -f "$study/runtime/S1_RUNNING.txt" ]]; then phase=S1
    else phase=GATE
    fi
    values=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits)
    echo "$(date --iso-8601=seconds),$phase,$values"
    sleep 5
  done
) > "$study/runtime/gpu_monitor.csv" 2> "$study/runtime/gpu_monitor.stderr" &
monitor_pid=$!
echo "$monitor_pid" > "$study/runtime/gpu_monitor.pid"
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

run_one() {
  local id=$1 config=$2 prefix=$3
  if compgen -G "${prefix}_*" >/dev/null; then
    echo "refusing existing run prefix: ${prefix}_*" >&2
    return 10
  fi
  printf 'id=%s\nconfig=%s\nstarted=%s\n' "$id" "$config" "$(date --iso-8601=seconds)" \
    > "$study/runtime/${id}_RUNNING.txt"
  CUDA_VISIBLE_DEVICES=0 "$py" train.py \
    --config "$config" \
    --checkpoint "$prefix" \
    --seed 0

  local run_dir
  run_dir=$(find "$(dirname "$prefix")" -maxdepth 1 -type d \
    -name "$(basename "$prefix")_*" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  [[ -n "$run_dir" ]] || { echo "run directory not found for $id" >&2; return 11; }
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
print(run)
PY
  for checkpoint_name in best_epoch.bin best_ema_epoch.bin; do
    local label=${checkpoint_name%.bin}
    CUDA_VISIBLE_DEVICES=0 "$py" train.py \
      --config "$config" \
      --evaluate "$run_dir/$checkpoint_name" \
      --checkpoint "$verification/${id}_${label}" \
      --seed 0 > "$run_dir/verification_${label}.stdout" 2>&1
  done
  printf 'id=%s\nrun_dir=%s\ncompleted=%s\n' "$id" "$run_dir" "$(date --iso-8601=seconds)" \
    > "$study/runtime/${id}_COMPLETE.txt"
  completed_run_dir=$run_dir
}

completed_run_dir=
run_one S1 \
  configs/pose3d/graph_posemamba_h36m_w128_d20_scale_80e.yaml \
  "$root/S1_w128_d20_seed0"
s1_run=$completed_run_dir
"$py" tools/summarize_scale_models.py --s1-run "$s1_run"

completed_run_dir=
run_one S2 \
  configs/pose3d/graph_posemamba_h36m_w256_d10_scale_80e.yaml \
  "$root/S2_w256_d10_seed0"
s2_run=$completed_run_dir
"$py" tools/summarize_scale_models.py --s1-run "$s1_run" --s2-run "$s2_run"

printf 'completed=%s\ns1_run=%s\ns2_run=%s\n' \
  "$(date --iso-8601=seconds)" "$s1_run" "$s2_run" \
  > "$study/SEQUENCE_COMPLETE.txt"
