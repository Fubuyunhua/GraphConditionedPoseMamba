#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_W128_D10_GT_PARALLEL_20260910
mkdir -p launch_logs verification
exec 9>verification/launch.lock
flock -n 9
test ! -e verification/formal_started
test ! -d runs/w128_d10_balanced_20260910
active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
test "$active" = 2004698
test "$(readlink /proc/2004698/cwd)" = /scratch/home/caiwei/GraphConditionedPoseMamba_W128_GT2D_SCRATCH_20260909
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
(( free_mib >= 16000 ))
nvidia-smi > verification/gpu_before.txt
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
"$py" -u tools/preflight_w128_d10_gt_parallel.py
touch verification/formal_started
set +e
"$py" -u train.py --config configs/pose3d/graph_posemamba_h36m_w128_d10_balanced_gt2d_80e.yaml --checkpoint runs/w128_d10_balanced_20260910/gt2d_seed0 --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
