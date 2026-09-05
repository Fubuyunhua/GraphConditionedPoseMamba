#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_REWIRED_20260906
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
exec 9>verification/rewired.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test ! -e verification/FORMAL_STARTED.json
test ! -d runs/paper_remaining_evidence
"$py" - <<'PY'
import json,math,pathlib,hashlib,torch,datetime
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
p=pathlib.Path('verification/rewired_b4.log').read_text()
b=json.loads(p[p.rfind('\n{')+1:])
assert b['real_data'] and b['batch_size']==4 and b['compiled']
assert math.isfinite(b['loss']) and b['iterations_per_second']>0
assert max(b['warmup_peak_reserved_mib'],b['measured_peak_reserved_mib'])<28672
assert not b['linear_warmup_enabled'] and b['optimizer_lr_after_steps']==0.0005
c=get_config('configs/pose3d/ablation_full_rewired_graph.yaml')
assert c.epochs==80 and c.seed==0 and c.graph_rewire_seed==3407
assert c.graph_topology_mode=='degree_preserving_rewired' and c.recurrence_scope=='independent'
assert not c.enable_linear_warmup and not c.honor_no_weight_decay and c.max_grad_norm==0
m=load_backbone(c); assert sum(p.numel() for p in m.parameters())==800083
manifest=json.loads(pathlib.Path('verification/source_manifest.json').read_text())
for n,h in manifest.items(): assert hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()==h,n
pathlib.Path('verification/effective_config.json').write_text(json.dumps(dict(c),indent=2,default=str))
identity={'status':'PASS','source_commit':'1bb0a15b5308c440fcbd9952e485f59e983a089b','scientific_implementation':'ad4e2fa66492737a3fcd88d9142a88731724f30b','dataset_sha256':'73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175','benchmark':b,'torch':torch.__version__,'cuda':torch.version.cuda,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':'python train.py --config configs/pose3d/ablation_full_rewired_graph.yaml --checkpoint runs/paper_remaining_evidence/full_rewired_graph_seed0 --seed 0'}
pathlib.Path('verification/PREFLIGHT_PASS.json').write_text(json.dumps(identity,indent=2))
pathlib.Path('verification/FORMAL_STARTED.json').write_text(json.dumps(identity,indent=2))
print(json.dumps(identity),flush=True)
PY
set +e
CUDA_VISIBLE_DEVICES=0 "$py" -u train.py --config configs/pose3d/ablation_full_rewired_graph.yaml --checkpoint runs/paper_remaining_evidence/full_rewired_graph_seed0 --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
