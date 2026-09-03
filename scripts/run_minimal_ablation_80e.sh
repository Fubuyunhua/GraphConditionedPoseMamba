#!/usr/bin/env bash
set -euo pipefail

repo=$(readlink -f "${1:-.}")
py=${PYTHON_BIN:-/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python}
study=.experiments/minimal_ablation_80e_20260903
root=runs/minimal_ablation_80e
verification=verification/minimal_ablation_80e

cd "$repo"
mkdir -p "$study/runtime" "$root" "$verification" launch_logs
exec 9>"$study/queue.lock"
flock -n 9 || { echo "minimal ablation queue lock is held" >&2; exit 3; }

[[ -f "$study/PRELAUNCH_PASS.json" ]] || {
  echo "missing $study/PRELAUNCH_PASS.json" >&2
  exit 4
}

if ! git diff --quiet -- lib/model/PoseMamba.py lib/utils/learning.py train.py \
  configs/pose3d/ablation_factorized_only.yaml \
  configs/pose3d/ablation_graph_feature.yaml; then
  echo "tracked experiment source is dirty" >&2
  exit 5
fi

active=$(ps -eo pid=,args= | awk -v repo="$repo" '$0 ~ /python/ && $0 ~ /train\.py/ && $0 ~ repo {print}')
[[ -z "$active" ]] || { echo "active target-repo training: $active" >&2; exit 6; }

dataset=data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl
expected=73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175
actual=$(sha256sum "$dataset" | awk '{print $1}')
[[ "$actual" == "$expected" ]] || { echo "dataset hash mismatch" >&2; exit 7; }

printf 'source_commit=%s\nstarted=%s\npython=%s\ndataset_sha256=%s\n' \
  "$(git rev-parse HEAD)" "$(date --iso-8601=seconds)" "$py" "$actual" \
  > "$study/runtime/queue_identity.txt"

run_one() {
  local id=$1 config=$2 prefix=$3
  if compgen -G "${prefix}_*" >/dev/null; then
    echo "refusing existing run prefix: ${prefix}_*" >&2
    return 8
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
  [[ -n "$run_dir" ]] || { echo "run directory not found for $id" >&2; return 9; }
  "$py" - "$run_dir" <<'PY'
import os
import sys
import torch

run = sys.argv[1]
for name in ("latest_epoch.bin", "latest_ema_epoch.bin", "best_epoch.bin", "best_ema_epoch.bin"):
    path = os.path.join(run, name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != 80 and name.startswith("latest"):
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
run_one \
  A1 \
  configs/pose3d/ablation_factorized_only.yaml \
  "$root/A1_factorized_only_seed0"
a1_run=$completed_run_dir
"$py" tools/summarize_minimal_ablation.py --a1-run "$a1_run"

completed_run_dir=
run_one \
  A2 \
  configs/pose3d/ablation_graph_feature.yaml \
  "$root/A2_graph_feature_seed0"
a2_run=$completed_run_dir
"$py" tools/summarize_minimal_ablation.py --a1-run "$a1_run" --a2-run "$a2_run"

printf 'completed=%s\na1_run=%s\na2_run=%s\n' \
  "$(date --iso-8601=seconds)" "$a1_run" "$a2_run" > "$study/SEQUENCE_COMPLETE.txt"
