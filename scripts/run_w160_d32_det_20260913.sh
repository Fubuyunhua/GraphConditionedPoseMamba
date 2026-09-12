#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_W160_D32_DET_20260913
mkdir -p verification launch_logs
exec 9>verification/launch.lock
flock -n 9
test ! -e verification/formal_started
test ! -d runs/w160_d32_max_20260913
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
(( free_mib >= 29000 ))
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
"$py" -u tools/preflight_w160_d32.py w160_d32_det2d
touch verification/formal_started
set +e
"$py" -u train.py --config configs/pose3d/graph_posemamba_h36m_w160_d32_max_det2d_80e.yaml --checkpoint runs/w160_d32_max_20260913/det2d_seed0 --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
