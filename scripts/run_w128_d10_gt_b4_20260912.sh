#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_W128_D10_GT_B4_20260912
mkdir -p verification launch_logs
exec 9>verification/launch.lock
flock -n 9
test ! -e verification/formal_started
test ! -d runs/w128_d10_gt_b4_20260912
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
(( free_mib >= 13000 ))
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
"$py" -u tools/preflight_conditioning_job.py gt_d10
touch verification/formal_started
set +e
"$py" -u train.py --config configs/pose3d/graph_posemamba_h36m_w128_d10_gt_b4_scratch_80e.yaml --checkpoint runs/w128_d10_gt_b4_20260912/seed0 --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
