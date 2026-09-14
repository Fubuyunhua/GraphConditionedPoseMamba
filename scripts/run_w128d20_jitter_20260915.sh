#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_W128D20_JITTER_FT_20260915
mkdir -p verification launch_logs
exec 9>verification/launch.lock
flock -n 9
test ! -e verification/formal_started
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
(( free_mib >= 20000 ))
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
config=configs/pose3d/ft_w128d20_detector_jitter_8e.yaml
"$py" -m unittest tests.test_layerwise_finetune tests.test_finetune_jitter
"$py" -u tools/preflight_w128d20_jitter.py
"$py" -u train.py --config "$config" --checkpoint runs/initial_eval --evaluate initializers/w128d20_e45_ema.bin --seed 0 > verification/initial_eval.log 2>&1
"$py" - <<'PY'
import json,re,pathlib
text=pathlib.Path('verification/initial_eval.log').read_text()
p1=float(re.findall(r'Protocol #1 Error \(MPJPE\):\s*([\d.]+)',text)[-1])
p2=float(re.findall(r'Protocol #2 Error \(P-MPJPE\):\s*([\d.]+)',text)[-1])
assert abs(p1-37.659301)<.02 and abs(p2-31.871747)<.02,(p1,p2)
pathlib.Path('verification/initial_eval.json').write_text(json.dumps({'p1':p1,'p2':p2,'status':'PASS'}))
print('INITIAL EVALUATION PASS',p1,p2,flush=True)
PY
touch verification/formal_started
set +e
"$py" -u train.py --config "$config" --checkpoint runs/w128d20_detector_jitter_20260915/seed0 --pretrained initializers --selection w128d20_e45_ema.bin --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
