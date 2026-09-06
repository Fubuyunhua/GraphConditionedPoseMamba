#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_H36M_GT2D_S_20260906
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
exec 9>verification/formal.lock
flock -n 9
test -f verification/GATES_PASS
test ! -e verification/FORMAL_STARTED.json
test ! -d runs/capacity_gt2d_20260906
"$py" - <<'PY'
import json,pathlib,math,subprocess,hashlib,datetime
from lib.utils.tools import get_config
c=get_config('configs/pose3d/graph_posemamba_h36m_w64_d8_gt2d_80e.yaml')
assert c.gt_2d and not c.no_eval and c.epochs==80 and c.batch_size==4
assert c.learning_rate==0.0005 and c.ema_decay==0.9998
assert not c.enable_linear_warmup and not c.honor_no_weight_decay
g=json.loads(pathlib.Path('verification/gt2d_protocol_gate.json').read_text())
t=pathlib.Path('verification/b4.log').read_text();b=json.loads(t[t.rfind('\n{')+1:])
assert g['status']=='PASS' and g['finite_gradients'] and g['parameters']==800083
assert b['real_data'] and b['compiled'] and b['batch_size']==4
assert math.isfinite(b['loss']) and b['iterations_per_second']>0
assert max(b['warmup_peak_reserved_mib'],b['measured_peak_reserved_mib'])<12000
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
assert set(active)<={'1136724'},active
if active:
 assert pathlib.Path('/proc/1136724/cwd').resolve()==pathlib.Path('/scratch/home/caiwei/GraphConditionedPoseMamba_MPI_FULL_20260906')
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>16000,free
data=pathlib.Path(c.data_root)/c.dt_file
h=hashlib.sha256()
with data.open('rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
assert h.hexdigest()=='73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175'
paths=['train.py','lib/data/dataset_motion_3d.py','lib/data/datareader_h36m.py','lib/model/PoseMamba.py','configs/pose3d/graph_posemamba_h36m_w64_d8_gt2d_80e.yaml']
p={'status':'PASS','time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_source':'7c60065dd0c20df383e9d23df80790f04279a135','delta':'S-GT config no_eval false plus preflight/launcher tools; scientific model unchanged','dataset_sha256':h.hexdigest(),'hashes':{n:hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest() for n in paths},'effective_config':dict(c),'protocol':g,'benchmark':b,'parallel_with':1136724 if active else None,'gpu_free_before_launch_mib':free,'selection':'best test-monitored EMA P1 first80; fixed80 secondary; GT known-input-xy evaluator'}
pathlib.Path('verification/PREFLIGHT_PASS.json').write_text(json.dumps(p,indent=2,default=str))
pathlib.Path('verification/FORMAL_STARTED.json').write_text(json.dumps(p,indent=2,default=str))
print(json.dumps(p,default=str),flush=True)
PY
set +e
CUDA_VISIBLE_DEVICES=0 "$py" -u train.py --config configs/pose3d/graph_posemamba_h36m_w64_d8_gt2d_80e.yaml --checkpoint runs/capacity_gt2d_20260906/s-gt2d_seed0 --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
