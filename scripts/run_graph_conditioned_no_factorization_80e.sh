#!/usr/bin/env bash
set -euo pipefail

repo=$(readlink -f "${1:-.}")
py=${PYTHON_BIN:-/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python}
study=.experiments/minimal_ablation_80e_20260903
config=configs/pose3d/ablation_graph_conditioned_no_factorization.yaml
prefix=runs/graph_posemamba/h36m/ablation_graph_conditioned_no_factorization_seed0
verification=verification/minimal_ablation_80e

cd "$repo"
mkdir -p "$study/runtime" "$(dirname "$prefix")" "$verification" launch_logs
exec 9>"$study/A4_queue.lock"
flock -n 9 || { echo "non-factorized ablation lock is held" >&2; exit 3; }

[[ -f "$study/A4_APPROVED" ]] || {
  echo "missing explicit approval marker: $study/A4_APPROVED" >&2
  exit 4
}
[[ -f "$study/A4_PRELAUNCH_PASS.json" ]] || {
  echo "missing regression preflight: $study/A4_PRELAUNCH_PASS.json" >&2
  exit 5
}
[[ -f "$study/SEQUENCE_COMPLETE.txt" ]] || {
  echo "A1/A2 sequence is not complete" >&2
  exit 6
}

if ! git diff --quiet -- lib/model/PoseMamba.py lib/model/mambablocks.py \
  lib/utils/learning.py train.py "$config"; then
  echo "tracked experiment source is dirty" >&2
  exit 7
fi

active=$(ps -eo pid=,args= | awk -v repo="$repo" \
  '$0 ~ /python/ && $0 ~ /train\.py/ && $0 ~ repo {print}')
[[ -z "$active" ]] || { echo "active target-repo training: $active" >&2; exit 8; }

if compgen -G "${prefix}_*" >/dev/null; then
  echo "refusing existing run prefix: ${prefix}_*" >&2
  exit 9
fi

dataset=data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl
expected=73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175
actual=$(sha256sum "$dataset" | awk '{print $1}')
[[ "$actual" == "$expected" ]] || { echo "dataset hash mismatch" >&2; exit 10; }

printf 'experiment=%s\nsource_commit=%s\nstarted=%s\npython=%s\ndataset_sha256=%s\n' \
  'Graph-Conditioned SSM w/o Factorization' "$(git rev-parse HEAD)" \
  "$(date --iso-8601=seconds)" "$py" "$actual" \
  > "$study/runtime/A4_RUNNING.txt"

CUDA_VISIBLE_DEVICES=0 "$py" train.py \
  --config "$config" \
  --checkpoint "$prefix" \
  --seed 0

run_dir=$(find "$(dirname "$prefix")" -maxdepth 1 -type d \
  -name "$(basename "$prefix")_*" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
[[ -n "$run_dir" ]] || { echo "run directory not found" >&2; exit 11; }

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
  label=${checkpoint_name%.bin}
  CUDA_VISIBLE_DEVICES=0 "$py" train.py \
    --config "$config" \
    --evaluate "$run_dir/$checkpoint_name" \
    --checkpoint "$verification/A4_${label}" \
    --seed 0 > "$run_dir/verification_${label}.stdout" 2>&1
done

a1_run=$(awk -F= '$1 == "a1_run" {print $2}' "$study/SEQUENCE_COMPLETE.txt")
a2_run=$(awk -F= '$1 == "a2_run" {print $2}' "$study/SEQUENCE_COMPLETE.txt")
"$py" tools/summarize_minimal_ablation.py \
  --a1-run "$a1_run" --a2-run "$a2_run" --a4-run "$run_dir"

printf 'experiment=%s\nrun_dir=%s\ncompleted=%s\n' \
  'Graph-Conditioned SSM w/o Factorization' "$run_dir" \
  "$(date --iso-8601=seconds)" > "$study/runtime/A4_COMPLETE.txt"
