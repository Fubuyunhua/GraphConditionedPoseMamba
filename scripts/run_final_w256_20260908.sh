#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_FINAL_W256_DET2D_20260908
mkdir -p verification launch_logs
exec 9>verification/launch.lock
flock -n 9
test ! -e verification/formal_started
old=/scratch/home/caiwei/GraphConditionedPoseMamba_FINAL_W128_DET2D_20260907
echo 'User requested immediate W128 stop; require its GPU process to exit'
! kill -0 1407974 2>/dev/null
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
bash scripts/preflight_final_w256_20260908.sh .
touch verification/formal_started
set +e
CUDA_VISIBLE_DEVICES=0 /scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python -u train.py --config configs/pose3d/graph_posemamba_h36m_final_w256_d16_det2d.yaml --checkpoint runs/final_two_sizes_20260906/w256_det2d_seed0 --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
