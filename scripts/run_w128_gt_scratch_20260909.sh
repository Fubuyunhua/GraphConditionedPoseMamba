#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_W128_GT2D_SCRATCH_20260909
mkdir -p launch_logs verification
exec 9>verification/launch.lock
flock -n 9
test ! -e verification/formal_started
test ! -d runs/w128_gt2d_scratch_20260909
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
(( free_mib >= 22000 ))
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
"$py" -u tools/preflight_w128_gt_scratch.py
touch verification/formal_started
set +e
"$py" -u train.py --config configs/pose3d/graph_posemamba_h36m_w128_gt2d_scratch_80e.yaml --checkpoint runs/w128_gt2d_scratch_20260909/seed0 --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
