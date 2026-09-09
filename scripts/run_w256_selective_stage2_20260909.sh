#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_W256_SELECTIVE_STAGE2_20260909
mkdir -p verification launch_logs
exec 9>verification/launch.lock
flock -n 9
test ! -e verification/formal_started
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
(( free_mib >= 16000 ))
nvidia-smi > verification/gpu_before.txt
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
config=configs/pose3d/graph_posemamba_h36m_w256_selective_stage2_8e.yaml
"$py" -m unittest tests.test_selective_finetune
"$py" -u tools/preflight_w256_stage2.py
"$py" -u train.py --config "$config" --checkpoint runs/initial_eval --evaluate initializers/stage1_best_ema.bin --seed 0 > verification/initial_eval.log 2>&1
"$py" - <<'PY'
import json,re,pathlib
text=pathlib.Path('verification/initial_eval.log').read_text()
p1=float(re.findall(r'Protocol #1 Error \(MPJPE\):\s*([\d.]+)',text)[-1])
p2=float(re.findall(r'Protocol #2 Error \(P-MPJPE\):\s*([\d.]+)',text)[-1])
assert abs(p1-37.416286)<.02 and abs(p2-31.507343)<.02,(p1,p2)
pathlib.Path('verification/initial_eval.json').write_text(json.dumps({'p1':p1,'p2':p2,'status':'PASS'}))
print('INITIAL EVALUATION PASS',p1,p2,flush=True)
PY
touch verification/formal_started
set +e
"$py" -u train.py --config "$config" --checkpoint runs/w256_selective_stage2_20260909/seed0 --pretrained initializers --selection stage1_best_ema.bin --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
