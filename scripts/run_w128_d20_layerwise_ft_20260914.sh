#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_W128_D20_LAYERWISE_FT_20260914
mkdir -p verification launch_logs
exec 9>verification/launch.lock
flock -n 9
test ! -e verification/formal_started
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
config=configs/pose3d/graph_posemamba_h36m_w128_d20_layerwise_ft_8e.yaml
"$py" -m unittest tests.test_layerwise_finetune
"$py" -u tools/preflight_w128_d20_layerwise_ft.py
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
"$py" -u train.py --config "$config" --checkpoint runs/w128_d20_layerwise_ft_20260914/seed0 --pretrained initializers --selection w128d20_e45_ema.bin --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
