#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_W256_BEST_GT2D_20260909
mkdir -p verification launch_logs
exec 9>verification/launch.lock
flock -n 9
test ! -e verification/formal_started
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
(( free_mib >= 28000 ))
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
config=configs/pose3d/graph_posemamba_h36m_w256_best_gt2d_30e.yaml
"$py" -u tools/preflight_w256_gt2d.py
"$py" -u train.py --config "$config" --checkpoint runs/initial_eval --evaluate initializers/gt2d_init_ema.bin --seed 0 > verification/initial_eval.log 2>&1
"$py" - <<'PY'
import json,re,pathlib
text=pathlib.Path('verification/initial_eval.log').read_text()
p1=float(re.findall(r'Protocol #1 Error \(MPJPE\):\s*([\d.]+)',text)[-1])
p2=float(re.findall(r'Protocol #2 Error \(P-MPJPE\):\s*([\d.]+)',text)[-1])
assert 0 < p1 < 1000 and 0 < p2 < 1000,(p1,p2)
pathlib.Path('verification/initial_eval.json').write_text(json.dumps({'p1':p1,'p2':p2,'status':'PASS'}))
print('INITIAL EVALUATION PASS',p1,p2,flush=True)
PY
touch verification/formal_started
set +e
"$py" -u train.py --config "$config" --checkpoint runs/w256_best_gt2d_20260909/seed0 --pretrained initializers --selection gt2d_init_ema.bin --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
